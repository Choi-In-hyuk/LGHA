"""
IDEA2 v2 Datasets.

DeltaDataset        (Stage 1): ResNet cache에서 연속 프레임 delta 반환.
LatentActionDataset (Stage 2/3): LiberoDataset + codebook index + context simulation.

Context simulation (Stage 2/3):
  - 50% 확률: context_vec = zeros  (첫 번째 primitive 상황)
  - 50% 확률: context_vec = 이전 step delta의 지수감쇠 합
    → 모델이 context 있을 때/없을 때 모두 학습됨
"""

import logging
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from hierarchical_vla.libero_bench.dataloader import LiberoDataset

log = logging.getLogger(__name__)


class DeltaDataset(Dataset):
    """
    Stage 1 VQ-VAE 학습용 데이터셋.

    ResNet cache에서 연속 프레임 쌍 delta 반환.
    delta = z_{t+1} - z_t  (ResNet scene 변화량)
    """

    def __init__(
        self,
        cache_dir: str,
        data_dir: str,
        demos_per_task: int = 50,
        step: int = 1,
    ):
        self.deltas = []

        file_list = sorted(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))

        for file in file_list:
            task_name = file.replace("_demo.hdf5", "")
            cache_path = os.path.join(cache_dir, f"{task_name}.pt")

            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"Cache not found: {cache_path}\n"
                    f"Run precompute_resnet.py first."
                )

            demo_cache = torch.load(cache_path, map_location="cpu")

            with h5py.File(os.path.join(data_dir, file), "r") as f:
                demo_keys = list(f["data"].keys())
                indices = np.argsort([int(k[5:]) for k in demo_keys])

            for i in indices[:demos_per_task]:
                demo_name = demo_keys[i]
                if demo_name not in demo_cache:
                    continue
                z = demo_cache[demo_name]  # (T, D)
                T = len(z)
                for t in range(0, T - step):
                    self.deltas.append(z[t + step] - z[t])

        self.deltas = torch.stack(self.deltas, dim=0)  # (N, D)
        log.info(f"DeltaDataset: {len(self.deltas)} delta samples")

    def __len__(self):
        return len(self.deltas)

    def __getitem__(self, idx):
        return self.deltas[idx]  # (D,)


class LatentActionDataset(LiberoDataset):
    """
    Stage 2/3 학습용 데이터셋.

    LiberoDataset 확장:
    - ResNet cache에서 z_scene 로드
    - codebook index 사전 계산 (배치)
    - context_vec 시뮬레이션

    __getitem__ 반환:
        obs, actions, mask, z_scene, codebook_idx, context_vec
    """

    def __init__(
        self,
        cache_dir: str,
        vqvae,                       # 학습된 VQVAEModel
        context_decay: float = 0.5,  # ContextEncoder decay
        context_dropout: float = 0.5, # context_vec = zeros 확률
        device: str = "cpu",
        **kwargs,
    ):
        log.info("LiberoDataset loading from HDF5...")
        super().__init__(device=device, **kwargs)
        log.info(f"Images loaded. {self.num_data} demos, {len(self.slices)} slices.")

        self.vqvae = vqvae
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

        self.cache_dir       = cache_dir
        self.context_decay   = context_decay
        self.context_dropout = context_dropout

        log.info("Loading ResNet cache...")
        self._load_z_cache()
        log.info("Precomputing codebook indices...")
        self._precompute_indices()
        log.info("Dataset ready.")

    def _load_z_cache(self):
        """ResNet cache 로드. per-demo z_scene 텐서 리스트."""
        file_list = sorted(
            f for f in os.listdir(self.data_dir) if f.endswith(".hdf5")
        )
        self.z_cache = []

        for file in file_list:
            task_name = file.replace("_demo.hdf5", "")
            cache_path = os.path.join(self.cache_dir, f"{task_name}.pt")

            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"Cache not found: {cache_path}\n"
                    f"Run precompute_resnet.py first."
                )

            demo_cache = torch.load(cache_path, map_location="cpu")

            with h5py.File(os.path.join(self.data_dir, file), "r") as f:
                demo_keys = list(f["data"].keys())
                indices = np.argsort([int(k[5:]) for k in demo_keys])

            for i in tqdm(
                indices[:self.demos_per_task],
                desc=f"  {task_name[:30]}",
                leave=False,
            ):
                demo_name = demo_keys[i]
                self.z_cache.append(demo_cache.get(demo_name, torch.zeros(1, 256)))

    def _precompute_indices(self):
        """모든 slice에 대해 codebook index 사전 계산 (배치 처리)."""
        z_starts = []
        z_ends   = []

        for (i, start, end) in self.slices:
            cache_len = len(self.z_cache[i])
            z_starts.append(self.z_cache[i][min(start, cache_len - 1)])
            z_ends.append(self.z_cache[i][min(end - 1, cache_len - 1)])

        z_starts_t = torch.stack(z_starts)
        z_ends_t   = torch.stack(z_ends)
        deltas     = z_ends_t - z_starts_t

        with torch.no_grad():
            indices = self.vqvae.encode_delta(deltas)

        self.z_starts          = list(z_starts_t)
        self.codebook_indices  = indices.tolist()

    def _simulate_context(self, demo_idx: int, start: int) -> torch.Tensor:
        """
        학습 시 compensation_vec 시뮬레이션.

        50% 확률: zeros (첫 primitive 상황)
        50% 확률: 같은 demo의 이전 delta들의 누적합
                  → z_scene - compensation ≈ 학습 분포

        학습 때도 inference와 동일한 방향: z_scene - compensation_vec
        """
        scene_dim = self.z_cache[demo_idx].shape[-1]

        if random.random() < self.context_dropout or start == 0:
            return torch.zeros(scene_dim)

        # 이전 chunk들의 delta 누적합 (단순합)
        z     = self.z_cache[demo_idx]
        compensation = torch.zeros(scene_dim)
        chunk = self.chunck_size

        t = start
        while t > 0:
            t_prev = max(0, t - chunk)
            delta  = z[min(t, len(z) - 1)] - z[min(t_prev, len(z) - 1)]
            compensation = compensation + delta
            t = t_prev
            if t == 0:
                break

        return compensation

    def __getitem__(self, idx):
        obs, act, mask = super().__getitem__(idx)

        i, start, end = self.slices[idx]

        z_scene     = self.z_starts[idx].float()
        codebook_idx = torch.tensor(self.codebook_indices[idx], dtype=torch.long)
        context_vec = self._simulate_context(i, start).float()

        return obs, act, mask, z_scene, codebook_idx, context_vec
