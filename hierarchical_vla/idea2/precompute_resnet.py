"""
ResNet scene embedding 사전 계산 스크립트.

SceneEncoder (frozen ResNet-18 + proj) 로 모든 demo frame의 z_scene을 미리 계산.
결과를 캐시로 저장 → Stage 1/2/3 학습 시 SceneEncoder forward 없이 빠른 학습.

중요: 여기서 저장한 SceneEncoder 가중치(scene_encoder.pt)를
      이후 모든 단계에서 동일하게 로드해야 z_scene 공간 일관성 보장.

Usage:
    python -m hierarchical_vla.idea2.precompute_resnet \\
        --data_dir /home/choi/LGHA/LIBERO/libero/datasets/libero_10_subset \\
        --cache_dir ./resnet_cache/libero_10_subset \\
        --demos_per_task 50
"""

import argparse
import logging
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .scene_encoder import SceneEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def precompute(
    data_dir: str,
    cache_dir: str,
    scene_emb_dim: int = 256,
    demos_per_task: int = 50,
    device: str = "cuda",
    batch_size: int = 64,
):
    os.makedirs(cache_dir, exist_ok=True)

    # SceneEncoder 생성 (frozen ResNet-18)
    encoder = SceneEncoder(output_dim=scene_emb_dim).to(device)
    encoder.eval()

    # 가중치 저장 → 이후 모든 단계에서 동일 파일 로드
    encoder_path = os.path.join(cache_dir, "scene_encoder.pt")
    if not os.path.exists(encoder_path):
        torch.save(encoder.state_dict(), encoder_path)
        log.info(f"Saved SceneEncoder weights: {encoder_path}")
    else:
        # 기존 가중치 로드 (재실행 시 일관성 유지)
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        log.info(f"Loaded existing SceneEncoder weights: {encoder_path}")

    file_list = sorted(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))
    log.info(f"Found {len(file_list)} hdf5 files")

    for file in tqdm(file_list, desc="Files"):
        task_name = file.replace("_demo.hdf5", "")
        cache_path = os.path.join(cache_dir, f"{task_name}.pt")

        if os.path.exists(cache_path):
            log.info(f"Skip (cached): {task_name}")
            continue

        with h5py.File(os.path.join(data_dir, file), "r") as f:
            demo_keys = list(f["data"].keys())
            indices = np.argsort([int(k[5:]) for k in demo_keys])

            demo_cache = {}

            for i in tqdm(indices[:demos_per_task], desc=f"  {task_name[:30]}", leave=False):
                demo_name = demo_keys[i]
                # agentview_rgb: (T, H, W, C) uint8
                frames = f["data"][demo_name]["obs"]["agentview_rgb"][:]
                T = len(frames)

                # (T, H, W, C) → (T, C, H, W) float [0,1]
                imgs = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0

                z_list = []
                with torch.no_grad():
                    for start in range(0, T, batch_size):
                        batch = imgs[start:start + batch_size].to(device)
                        z = encoder(batch)
                        z_list.append(z.cpu())

                demo_cache[demo_name] = torch.cat(z_list, dim=0)  # (T, scene_emb_dim)

        torch.save(demo_cache, cache_path)
        log.info(f"Saved: {cache_path}")

    log.info("Pre-computation complete.")
    log.info(f"SceneEncoder weights at: {encoder_path}")
    log.info("Pass this cache_dir to train_stage1/2/3 as --cache_dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str, required=True)
    parser.add_argument("--cache_dir",      type=str, required=True)
    parser.add_argument("--scene_emb_dim",  type=int, default=256)
    parser.add_argument("--demos_per_task", type=int, default=50)
    parser.add_argument("--device",         type=str, default="cuda")
    parser.add_argument("--batch_size",     type=int, default=64)
    args = parser.parse_args()

    precompute(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        scene_emb_dim=args.scene_emb_dim,
        demos_per_task=args.demos_per_task,
        device=args.device,
        batch_size=args.batch_size,
    )
