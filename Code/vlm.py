# vlm.py

from __future__ import annotations

from typing import List, Tuple

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import MAX_NEW_TOKENS

#모델 로드
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
dtype    = torch.float16 if torch.cuda.is_available() else torch.float32

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=dtype, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("model loaded:", MODEL_ID)


#추론 함수

def run_vlm(
    image_path: str,
    prompt: str,
    return_logprobs: bool = False,
) -> Tuple[str, List[float]]:
    """
    단일 이미지 + 텍스트 프롬프트를 VLM에 입력하고 생성 결과를 반환한다.

    Args:
        image_path      : 로컬 이미지 경로
        prompt          : 텍스트 프롬프트
        return_logprobs : True이면 생성 토큰의 log-prob 리스트도 반환

    Returns:
        (생성 텍스트, log_probs) 튜플.
        return_logprobs=False 이면 log_probs 는 빈 리스트.
    """
    image   = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    text_in = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text_in], images=[image], return_tensors="pt").to(
        model.device
    )

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_scores=True,
        )

    gen_ids  = out.sequences[:, inputs["input_ids"].shape[1] :]
    text_out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    log_probs: List[float] = []
    if return_logprobs and out.scores:
        for step_idx, step_scores in enumerate(out.scores):
            token_id = gen_ids[0, step_idx].item()
            log_probs.append(
                torch.log_softmax(step_scores[0], dim=-1)[token_id].item()
            )

    return text_out, log_probs
