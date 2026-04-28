"""
V-JEPA 임베딩 사전 계산 스크립트.

학습 전 한 번만 실행하면 됨.
모든 demo frame의 z_scene을 미리 계산해 캐시로 저장.
학습 중엔 캐시를 읽기만 해서 V-JEPA forward 없이 빠르게 학습 가능.

Usage:
    python -m models.precompute_vjepa \
        --data_dir /home/choi/LGHA/LIBERO/libero/datasets/libero_10_subset \
        --cache_dir ./vjepa_cache/libero_10_subset \
        --demos_per_task 50
"""

import argparse
import os
import logging
import numpy as np
import torch
import h5py
from tqdm import tqdm

from .vjepa_encoder import VJEPASceneEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def precompute(
    data_dir: str,
    cache_dir: str,
    model_name: str = "facebook/vjepa2-vitg-fpc64-256",
    scene_emb_dim: int = 256,
    n_frames: int = 1,
    demos_per_task: int = 50,
    device: str = "cuda",
    batch_size: int = 32,
):
    os.makedirs(cache_dir, exist_ok=True)

    # 완전히 frozen으로 로드 (fine-tune 없음, 사전계산용)
    encoder = VJEPASceneEncoder(
        model_name=model_name,
        output_dim=scene_emb_dim,
        n_finetune_layers=0,   # 전체 frozen
        n_frames=n_frames,
    ).to(device)
    encoder.eval()

    file_list = sorted(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))
    log.info(f"Found {len(file_list)} hdf5 files")

    for file in tqdm(file_list, desc="Files"):
        task_name = file.replace("_demo.hdf5", "")
        cache_path = os.path.join(cache_dir, f"{task_name}.pt")

        if os.path.exists(cache_path):
            log.info(f"Skip (cached): {task_name}")
            continue

        f = h5py.File(os.path.join(data_dir, file), "r")
        demo_keys = list(f["data"].keys())
        indices = np.argsort([int(k[5:]) for k in demo_keys])

        demo_cache = {}  # demo_name → z_scene tensor (T, scene_emb_dim)

        total_frames = sum(
            f["data"][demo_keys[i]].attrs["num_samples"]
            for i in indices[:demos_per_task]
        )
        with tqdm(total=total_frames, desc=f"  {task_name[:30]}", leave=False) as pbar:
            for i in indices[:demos_per_task]:
                demo_name = demo_keys[i]
                frames = f["data"][demo_name]["obs"]["agentview_rgb"][:]  # (T, H, W, C)
                T = len(frames)

                imgs = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0

                z_list = []
                with torch.no_grad():
                    for start in range(0, T, batch_size):
                        batch = imgs[start:start + batch_size].to(device)
                        z = encoder(batch)
                        z_list.append(z.cpu())
                        pbar.update(len(batch))

                demo_cache[demo_name] = torch.cat(z_list, dim=0)

        f.close()
        torch.save(demo_cache, cache_path)
        log.info(f"Saved: {cache_path}")

    log.info("Pre-computation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/vjepa2-vitg-fpc64-256")
    parser.add_argument("--scene_emb_dim", type=int, default=256)
    parser.add_argument("--n_frames", type=int, default=1)
    parser.add_argument("--demos_per_task", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    precompute(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        scene_emb_dim=args.scene_emb_dim,
        n_frames=args.n_frames,
        demos_per_task=args.demos_per_task,
        device=args.device,
        batch_size=args.batch_size,
    )
