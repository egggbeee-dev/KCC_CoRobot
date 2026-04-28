# p2p_ablation.py
#
# Ablation Study: offer / negotiation / hq 각 컴포넌트 기여도 측정
#
# p2p_main.run()의 플래그만 바꿔서 실행
# 측정: PT (elapsed time), TC (token cost) 만 측정
#
# 조건:
#   Full P2P        : use_offer=True,  use_negotiation=True,  use_human_query=True
#   w/o Offer       : use_offer=False, use_negotiation=True,  use_human_query=True
#   w/o Negotiate   : use_offer=True,  use_negotiation=False, use_human_query=True
#   w/o HQ          : use_offer=True,  use_negotiation=True,  use_human_query=False
#
# 실행:
#   from p2p_ablation import run_ablation_study
#   run_ablation_study(
#       task_id     = "task_003",
#       image_pairs = [(img_a, img_b), ...],
#   )

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from IPython.display import display

from p2p_main import get_task, run
from p2p_tracker import tracker
from p2p_utils import _banner


ABLATION_CONDITIONS: List[Tuple[str, Dict]] = [
    ("Full P2P",      dict(use_offer=True,  use_negotiation=True,  use_human_query=True)),
    ("w/o Offer",     dict(use_offer=False, use_negotiation=True,  use_human_query=True)),
    ("w/o Negotiate", dict(use_offer=True,  use_negotiation=False, use_human_query=True)),
    ("w/o HQ",        dict(use_offer=True,  use_negotiation=True,  use_human_query=False)),
]


def _save_result(result: Dict, condition: str, pt: float, tc: int, run_idx: int):
    save_dir = Path("/content/KCC_CoRobot/results")
    save_dir.mkdir(exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    cond  = condition.replace(" ", "_").replace("/", "")
    fname = save_dir / f"ablation_{result['task_id']}_{cond}_run{run_idx}_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({**result, "condition": condition, "pt": pt, "tc": tc},
                  f, ensure_ascii=False, indent=2)
    print(f"  → 저장: {fname}")


def run_ablation_study(
    task_id:     str,
    image_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Args:
        task_id     : 실험 태스크 ID
        image_pairs : [(img_a, img_b), ...] 이미지 페어 리스트
    """
    SEP = "═" * 68
    print(SEP)
    print(f"  ABLATION STUDY  |  task={task_id}  |  N={len(image_pairs)}")
    print(SEP)

    all_rows: Dict[str, List[Dict]] = {cond: [] for cond, _ in ABLATION_CONDITIONS}

    for run_idx, (img_a, img_b) in enumerate(image_pairs, 1):
        print(f"\n{'━'*68}")
        print(f"  [Run {run_idx}/{len(image_pairs)}]")
        print(f"  img_a: {img_a}")
        print(f"  img_b: {img_b}")
        print(f"{'━'*68}")

        for condition, flags in ABLATION_CONDITIONS:
            _banner(f"ABLATION — {condition}")
            print(f"  Flags: {flags}")

            tracker.start()
            try:
                result = run(
                    task_id = task_id,
                    img_a   = img_a,
                    img_b   = img_b,
                    label   = condition,
                    **flags,
                )
            except Exception as e:
                print(f"  [ERROR] {condition}: {e}")
                tracker.stop()
                all_rows[condition].append({"pt": 0.0, "tc": 0})
                continue
            tracker.stop()

            pt = tracker.elapsed
            tc = tracker.total_tokens
            print(tracker.summary(condition))
            _save_result(result, condition, pt, tc, run_idx)
            all_rows[condition].append({"pt": pt, "tc": tc})

    # ── 결과 테이블 ──────────────────────────────────────────────────────────
    final_rows = []
    for condition, _ in ABLATION_CONDITIONS:
        rows = all_rows[condition]
        final_rows.append({
            "Condition": condition,
            "PT(s)":     round(float(np.mean([r["pt"] for r in rows])), 2),
            "TC":        int(np.mean([r["tc"] for r in rows])),
        })

    df = pd.DataFrame(final_rows)[["Condition", "PT(s)", "TC"]]

    print("\n" + "█" * 68)
    print("  Table. Ablation Study — PT / TC")
    print("█" * 68)
    display(
        df.style
          .hide(axis="index")
          .format({"PT(s)": "{:.2f}", "TC": "{:,}"})
          .set_properties(**{"text-align": "center"})
    )
    print("\n[Markdown]")
    print(df.to_markdown(index=False))
    print("\n※ 플랜 품질은 위 자연어 출력을 통해 확인하세요.")

    return df
