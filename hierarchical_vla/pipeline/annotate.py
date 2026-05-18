"""
오프라인 annotation 스크립트.

LIBERO HDF5 데이터셋을 순회하면서 각 demo에 대해:
  1. Qwen2-VL → loc_tokens (이미지 기반 물체 위치, stride마다)
  2. Rule-based → phase_labels (gripper_states + ee_pos 기반, 자동)

{suite}_annotations.pkl로 저장.

개선 사항:
  - action_type: task 이름 기반 → phase 레이블 (5단계: reach/grasp/transport/place/release)
  - phase_labels: gripper + ee_pos로 자동 생성, 수동 레이블 불필요
  - loc_tokens: Qwen2-VL 유지 (annotation용; eval은 GT 투영 사용)

사용법:
  python -m hierarchical_vla.pipeline.annotate \
      --data_dir /home/choi/libero_object \
      --stride 5 \
      --model Qwen/Qwen2-VL-7B-Instruct
"""

import argparse
import logging
import os
import pickle

import h5py
import numpy as np

from .qwen_orchestrator import QwenOrchestrator
from .gt_localizer import label_phases_from_demo, NUM_PHASES, PHASE_NAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def task_name_to_instruction(task_name: str) -> str:
    """HDF5 파일명 → 자연어 명령으로 변환."""
    name = task_name.replace("_demo", "").replace("_", " ")
    # KITCHEN_SCENE1_... 형식이면 scene prefix 제거
    parts = name.split()
    for i, p in enumerate(parts):
        if p.lower().startswith("scene") and i > 0:
            name = " ".join(parts[i + 1:])
            break
    return name


def annotate_suite(
    data_dir: str,
    output_path: str,
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    stride: int = 5,
    device: str = "cuda",
    demos_per_task: int = None,
):
    """
    저장 형식:
    {
        "<task_file>_<demo_key>": {
            "action_type":  int,          # task 기반 (레거시, 하위 호환용)
            "phase_labels": List[int],    # [T] per-frame phase (0-4) ← NEW
            "loc_tokens":   List[int],    # [T] Qwen2-VL 기반 object 위치
            "instruction":  str,
        },
    }

    phase_labels:
      0=reach, 1=grasp, 2=transport, 3=place, 4=release
      gripper_states + ee_pos에서 자동 생성 (수동 레이블링 불필요)
    """
    orchestrator = QwenOrchestrator(model_id=model_id, device=device)

    annotations  = {}
    hdf5_files   = sorted(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))
    log.info(f"Found {len(hdf5_files)} HDF5 files in {data_dir}")

    for hdf5_file in hdf5_files:
        task_name   = os.path.splitext(hdf5_file)[0]
        instruction = task_name_to_instruction(task_name)
        log.info(f"[{task_name}]\n  instruction: '{instruction}'")

        fpath = os.path.join(data_dir, hdf5_file)
        with h5py.File(fpath, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k[5:]))
            if demos_per_task is not None:
                demo_keys = demo_keys[:demos_per_task]

            for demo_key in demo_keys:
                demo   = f["data"][demo_key]
                frames = demo["obs"]["agentview_rgb"][:]        # [T, H, W, C]
                gripper_states = demo["obs"]["gripper_states"][:] # [T, 2]
                ee_pos         = demo["obs"]["ee_pos"][:]          # [T, 3]

                # ── loc_tokens: Qwen2-VL (변경 없음) ─────────────────────
                action_type, loc_tokens = orchestrator.annotate_frames(
                    frames, instruction, stride=stride
                )

                # ── phase_labels: rule-based 자동 생성 ───────────────────
                phase_labels = label_phases_from_demo(gripper_states, ee_pos)

                # phase 분포 로그
                unique, counts = np.unique(phase_labels, return_counts=True)
                dist = {PHASE_NAMES[p]: int(c) for p, c in zip(unique, counts)}

                ann_key = f"{task_name}_{demo_key}"
                annotations[ann_key] = {
                    "action_type":  action_type,
                    "phase_labels": phase_labels.tolist(),
                    "loc_tokens":   loc_tokens,
                    "instruction":  instruction,
                }
                log.info(f"  {ann_key}: action={action_type}, phases={dist}, frames={len(loc_tokens)}")

    with open(output_path, "wb") as f:
        pickle.dump(annotations, f)

    log.info(f"Saved {len(annotations)} annotations → {output_path}")
    return annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      required=True)
    parser.add_argument("--output_dir",    default=None)
    parser.add_argument("--model",         default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--stride",        type=int, default=5)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--demos_per_task",type=int, default=None)
    args = parser.parse_args()

    output_dir  = args.output_dir or args.data_dir
    suite_name  = os.path.basename(args.data_dir.rstrip("/"))
    output_path = os.path.join(output_dir, f"{suite_name}_annotations.pkl")

    annotate_suite(
        data_dir=args.data_dir,
        output_path=output_path,
        model_id=args.model,
        stride=args.stride,
        device=args.device,
        demos_per_task=args.demos_per_task,
    )


if __name__ == "__main__":
    main()
