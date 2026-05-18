"""
Qwen2-VL 파인튜닝용 데이터 생성 스크립트.

LIBERO HDF5 demo의 저장된 simulator states를 복원해서
GT obj_loc_token + target_loc_token을 추출한다.

출력 형식 (JSONL):
{
  "image": "<base64 PNG>",
  "instruction": "pick up the alphabet soup and place it in the basket",
  "response": "object: 342 target: 615"
}

사용법:
  python -m hierarchical_vla.pipeline.generate_finetune_data \
      --data_dir /home/choi/libero_object \
      --output_path /home/choi/libero_object/qwen_finetune.jsonl \
      --stride 10 \
      --demos_per_task 50
"""

import argparse
import base64
import json
import logging
import os
import sys

import h5py
import numpy as np
from PIL import Image
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LIBERO"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

from .gt_localizer import get_gt_loc_token, get_gt_target_loc_token, extract_object_name


def image_to_base64(img_array: np.ndarray) -> str:
    """[H, W, C] uint8 RGB → base64 PNG 문자열."""
    pil = Image.fromarray(img_array.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def task_name_to_instruction(task_name: str) -> str:
    """HDF5 파일명 → 자연어 명령."""
    name = task_name.replace("_demo", "").replace("_", " ")
    parts = name.split()
    for i, p in enumerate(parts):
        if p.lower().startswith("scene") and i > 0:
            name = " ".join(parts[i + 1:])
            break
    return name.strip()


def generate_finetune_data(
    data_dir: str,
    output_path: str,
    suite_name: str = None,
    stride: int = 10,
    demos_per_task: int = 50,
    img_h: int = 128,
    img_w: int = 128,
    camera: str = "agentview",
):
    """
    LIBERO HDF5 데이터셋을 순회하며 Qwen2-VL 파인튜닝 데이터를 생성.

    HDF5의 저장된 'states'를 MuJoCo simulator에 복원해
    GT obj_loc + target_loc을 정확히 계산한다.
    """
    import torch as _torch
    _orig_load = _torch.load
    _torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    # suite_name 자동 추론
    if suite_name is None:
        suite_name = os.path.basename(data_dir.rstrip("/"))

    bm_dict  = benchmark.get_benchmark_dict()
    bm       = bm_dict[suite_name]()
    num_tasks = bm.get_num_tasks()

    hdf5_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))
    log.info(f"Found {len(hdf5_files)} HDF5 files | suite={suite_name} | stride={stride}")

    records = []
    task_map = {bm.get_task(i).name: i for i in range(num_tasks)}

    for hdf5_file in hdf5_files:
        task_name   = os.path.splitext(hdf5_file)[0]
        instruction = task_name_to_instruction(task_name)
        fpath       = os.path.join(data_dir, hdf5_file)

        # HDF5 파일명의 _demo 접미사 제거 후 벤치마크 매칭
        task_name_clean = task_name.removesuffix("_demo")
        task_id = None
        for i in range(num_tasks):
            if bm.get_task(i).name == task_name_clean:
                task_id = i
                break
        if task_id is None:
            log.warning(f"Task '{task_name_clean}' not found in benchmark. Skipping.")
            continue
        task_name = task_name_clean   # 이후 GT 함수에는 clean name 사용

        task = bm.get_task(task_id)
        env_args = {
            "bddl_file_name": os.path.join(
                os.path.dirname(__file__), "..", "..", "LIBERO",
                "libero", "libero", "bddl_files",
                suite_name, f"{task.name}.bddl"
            ),
            "camera_names": [camera],
            "camera_heights": img_h,
            "camera_widths":  img_w,
            "render_camera":  camera,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()

        obj_name = extract_object_name(task_name)
        log.info(f"[{task_name}] obj='{obj_name}' instruction='{instruction}'")

        with h5py.File(fpath, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k[5:]))
            demo_keys = demo_keys[:demos_per_task]

            for demo_key in demo_keys:
                demo   = f["data"][demo_key]
                T      = int(demo.attrs.get("num_samples", len(demo["actions"])))
                images = demo["obs"][f"{camera}_rgb"][:]   # [T, H, W, C]

                if "states" not in demo:
                    log.warning(f"  {demo_key}: 'states' 없음. Skipping.")
                    continue

                states = demo["states"][:]   # [T, state_dim]

                step_count = 0
                for t in range(0, T, stride):
                    # MuJoCo 상태 복원
                    env.sim.set_state_from_flattened(states[t])
                    env.sim.forward()

                    obj_loc    = get_gt_loc_token(env, task_name, camera, img_h, img_w)
                    target_loc = get_gt_target_loc_token(env, task_name, camera, img_h, img_w)

                    img_b64 = image_to_base64(images[t])
                    records.append({
                        "image":       img_b64,
                        "instruction": instruction,
                        "response":    f"object: {obj_loc} target: {target_loc}",
                        "obj_loc":     obj_loc,
                        "target_loc":  target_loc,
                        "task_name":   task_name,
                        "demo_key":    demo_key,
                        "frame":       t,
                    })
                    step_count += 1

                log.info(f"  {demo_key}: {step_count} samples")

        env.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"\nTotal {len(records)} samples → {output_path}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       required=True)
    parser.add_argument("--output_path",    required=True)
    parser.add_argument("--suite_name",     default=None)
    parser.add_argument("--stride",         type=int, default=10)
    parser.add_argument("--demos_per_task", type=int, default=50)
    parser.add_argument("--camera",         default="agentview")
    args = parser.parse_args()

    generate_finetune_data(
        data_dir=args.data_dir,
        output_path=args.output_path,
        suite_name=args.suite_name,
        stride=args.stride,
        demos_per_task=args.demos_per_task,
        camera=args.camera,
    )


if __name__ == "__main__":
    main()
