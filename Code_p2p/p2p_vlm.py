# ══════════════════════════════════════════════════════════════════════════════
# vlm.py
# VLM 백엔드 선택: Qwen2.5-VL (로컬) 또는 GPT-4o (OpenAI API)
#
# 사용법:
#   환경변수로 백엔드 선택
#   VLM_BACKEND=qwen    → Qwen2.5-VL-7B (기본값, GPU 필요)
#   VLM_BACKEND=openai  → GPT-4o Vision (OpenAI API 키 필요)
#
#   코랩에서 OpenAI로 전환:
#     import os
#     os.environ["VLM_BACKEND"] = "openai"
#     os.environ["OPENAI_API_KEY"] = "sk-..."
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import base64
import os
import threading
from typing import List, Tuple

from p2p_config import MAX_NEW_TOKENS


# ── 토큰 사용량 추적 ──────────────────────────────────────────────────────────
# _last_usage  : 마지막 단일 호출의 토큰 (하위 호환용)
# _total_usage : 실험 단위 누적 토큰 (thread-safe, p2p_tracker에서 사용)
_last_usage:  dict            = {"prompt_tokens": 0, "completion_tokens": 0}
_usage_lock:  threading.Lock  = threading.Lock()
_total_usage: dict            = {"prompt_tokens": 0, "completion_tokens": 0}

# ── 백엔드 선택 ───────────────────────────────────────────────────────────────
VLM_BACKEND = os.environ.get("VLM_BACKEND", "qwen").lower()

# ──────────────────────────────────────────────────────────────────────────────
# Qwen 백엔드
# ──────────────────────────────────────────────────────────────────────────────

_qwen_model     = None
_qwen_processor = None

def _load_qwen():
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return

    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype    = torch.float16 if torch.cuda.is_available() else torch.float32

    _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto"
    )
    _qwen_processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("model loaded:", MODEL_ID)


def _run_qwen(image_path: str, prompt: str, return_logprobs: bool = False) -> Tuple[str, List[float]]:
    import torch
    from PIL import Image, ImageOps

    _load_qwen()

    image    = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text_in  = _qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = _qwen_processor(text=[text_in], images=[image], return_tensors="pt").to(_qwen_model.device)

    with torch.no_grad():
        out = _qwen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_scores=True,
        )

    gen_ids  = out.sequences[:, inputs["input_ids"].shape[1]:]
    text_out = _qwen_processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    log_probs: List[float] = []
    if return_logprobs and out.scores:
        for step_idx, step_scores in enumerate(out.scores):
            token_id = gen_ids[0, step_idx].item()
            log_probs.append(
                torch.log_softmax(step_scores[0], dim=-1)[token_id].item()
            )

    return text_out, log_probs


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI GPT-4o 백엔드
# ──────────────────────────────────────────────────────────────────────────────

_openai_client = None

def _load_openai():
    global _openai_client
    if _openai_client is not None:
        return

    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "  os.environ['OPENAI_API_KEY'] = 'sk-...' 로 설정하세요."
        )
    _openai_client = OpenAI(api_key=api_key)
    print("OpenAI client loaded: gpt-4o")


def _run_openai(image_path: str, prompt: str, return_logprobs: bool = False) -> Tuple[str, List[float]]:
    _load_openai()

    ext  = image_path.rsplit(".", 1)[-1].lower()
    mime = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "webp": "image/webp",
    }.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    kwargs = {
        "model": "gpt-4o",
        "max_tokens": MAX_NEW_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }

    if return_logprobs:
        kwargs["logprobs"]     = True
        kwargs["top_logprobs"] = 1

    response = _openai_client.chat.completions.create(**kwargs)

    # 마지막 호출 토큰 기록 (하위 호환)
    _last_usage["prompt_tokens"]     = response.usage.prompt_tokens
    _last_usage["completion_tokens"] = response.usage.completion_tokens

    # 실험 단위 누적 토큰 기록 (thread-safe)
    with _usage_lock:
        _total_usage["prompt_tokens"]     += response.usage.prompt_tokens
        _total_usage["completion_tokens"] += response.usage.completion_tokens

    text_out  = response.choices[0].message.content.strip()
    log_probs = []
    if return_logprobs and response.choices[0].logprobs:
        log_probs = [t.logprob for t in response.choices[0].logprobs.content]

    return text_out, log_probs


# ──────────────────────────────────────────────────────────────────────────────
# 통합 인터페이스
# ──────────────────────────────────────────────────────────────────────────────

def run_vlm(
    image_path: str,
    prompt: str,
    return_logprobs: bool = False,
) -> Tuple[str, List[float]]:
    """
    VLM 추론 통합 인터페이스.
    VLM_BACKEND 환경변수에 따라 Qwen 또는 GPT-4o를 사용한다.

    Args:
        image_path      : 로컬 이미지 경로
        prompt          : 텍스트 프롬프트
        return_logprobs : True이면 token log-prob 리스트도 반환

    Returns:
        (생성 텍스트, log_probs) 튜플
    """
    backend = os.environ.get("VLM_BACKEND", VLM_BACKEND).lower()

    if backend == "openai":
        return _run_openai(image_path, prompt, return_logprobs)
    else:
        return _run_qwen(image_path, prompt, return_logprobs)


def get_backend() -> str:
    """현재 사용 중인 백엔드 이름 반환."""
    return os.environ.get("VLM_BACKEND", VLM_BACKEND).lower()
