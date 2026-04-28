"""
WorldModelDataset

LiberoDataset를 확장해 goal frame (demo 마지막 프레임)을 추가 반환.

학습 시:
  - obs_t:      현재 observation
  - goal_frame: 같은 demo의 마지막 agentview 프레임  ← NEW
  - actions:    action chunk
  - lang_emb:   task embedding
"""

import torch
import numpy as np

from hierarchical_vla.libero_bench.dataloader import LiberoDataset


class WorldModelDataset(LiberoDataset):
    """LiberoDataset + goal_frame (demo 마지막 프레임)."""

    def __getitem__(self, idx):
        obs, act, mask = super().__getitem__(idx)

        i, start, end = self.slices[idx]

        # demo의 마지막 유효 프레임
        T = int(self.masks[i].sum().item())
        goal_rgb = self.agentview_rgb[i][T - 1]  # (H, W, C), uint8

        goal_frame = (
            torch.from_numpy(goal_rgb)
            .float()
            .permute(2, 0, 1)    # (C, H, W)
            / 255.0
        )

        obs["goal_frame"] = goal_frame
        return obs, act, mask
