# p2p_tracker.py
#
# PT (Planning Time) + TC (Token Cost) 측정 모듈
#
# 사용법:
#   from p2p_tracker import tracker
#   tracker.start()
#   ... 실험 실행 ...
#   tracker.stop()
#   print(tracker.summary())
#
# p2p_vlm.py에 아래 두 줄이 추가되어 있어야 합니다:
#   _last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}
#   # _run_openai 내부 response 생성 직후:
#   _last_usage["prompt_tokens"]     = response.usage.prompt_tokens
#   _last_usage["completion_tokens"] = response.usage.completion_tokens

from __future__ import annotations
import time
import p2p_vlm

class ExperimentTracker:
    """실험 단위 PT / TC 누적 트래커."""

    def __init__(self):
        self._start: float | None = None
        self.elapsed: float       = 0.0
        self.input_tokens: int    = 0
        self.output_tokens: int   = 0

    # ── 타이머 ──────────────────────────────────────────────────────────────
    def start(self):
        self._start = time.time()
        self.elapsed       = 0.0
        self.input_tokens  = 0
        self.output_tokens = 0
        # run_vlm 패치 활성화
        p2p_vlm.run_vlm = self._patched_run_vlm

    def stop(self):
        if self._start is not None:
            self.elapsed = round(time.time() - self._start, 2)
        # 원본 복원
        p2p_vlm.run_vlm = _original_run_vlm

    # ── 토큰 누적 ───────────────────────────────────────────────────────────
    def _patched_run_vlm(self, image_path, prompt, return_logprobs=False):
        raw, logp = _original_run_vlm(image_path, prompt, return_logprobs)
        usage = getattr(p2p_vlm, "_last_usage", {})
        self.input_tokens  += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        return raw, logp

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    # ── 요약 출력 ───────────────────────────────────────────────────────────
    def summary(self, label: str = "") -> str:
        tag = f"[{label}] " if label else ""
        return (
            f"  {tag}PT = {self.elapsed:.1f}s  |  "
            f"TC = {self.total_tokens:,} tokens "
            f"(in={self.input_tokens:,} / out={self.output_tokens:,})"
        )

    def as_dict(self) -> dict:
        return {
            "pt":             self.elapsed,
            "tc":             self.total_tokens,
            "input_tokens":   self.input_tokens,
            "output_tokens":  self.output_tokens,
        }


# 원본 보관
_original_run_vlm = p2p_vlm.run_vlm

# 싱글턴
tracker = ExperimentTracker()
