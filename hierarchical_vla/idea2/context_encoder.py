"""
ContextEncoder: delta_history → compensation_vec.

핵심 아이디어:
  z_scene_now - cumulative_delta ≈ z_scene_train

이전 primitive들이 환경에 만든 누적 변화량을 계산.
ActionDecoder에서 z_scene에서 빼면 학습 분포에 가까워짐.

예시:
  primitive 1: 파란접시 치움 → delta_1 = z_after - z_before
  primitive 2 시작:
    compensation = delta_1
    z_compensated = z_scene_now - compensation
    → VLA 입장: "파란접시 있던 때처럼" 보정됨
"""

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    delta_history → compensation_vec (scene_dim,).

    단순 누적합: compensation = delta_1 + delta_2 + ...
    학습 파라미터 없음.

    사용:
        z_compensated = z_scene - compensation_vec
        ActionDecoder(codebook[k], z_compensated)
    """

    def __init__(self, scene_dim: int = 256):
        super().__init__()
        self.scene_dim = scene_dim

    def forward(self, delta_history: list) -> torch.Tensor:
        """
        Args:
            delta_history: list of Tensor (scene_dim,), oldest → newest

        Returns:
            compensation_vec: (scene_dim,) — 비어있으면 zero
        """
        if not delta_history:
            return torch.zeros(self.scene_dim)

        # 누적 delta 합산: 이 값을 z_scene에서 빼면 학습 분포 복원
        compensation = torch.zeros(self.scene_dim)
        for delta in delta_history:
            compensation = compensation + delta.float().cpu()

        return compensation  # (scene_dim,)

    def compensate(
        self,
        z_scene: torch.Tensor,
        delta_history: list,
    ) -> torch.Tensor:
        """
        z_scene에서 누적 delta를 빼서 보정된 z_scene 반환.

        Args:
            z_scene:       (B, scene_dim) or (scene_dim,)
            delta_history: list of Tensor (scene_dim,)

        Returns:
            z_compensated: z_scene과 동일한 shape
        """
        compensation = self.forward(delta_history).to(z_scene.device)

        if z_scene.dim() == 2:
            compensation = compensation.unsqueeze(0)  # (1, scene_dim)

        return z_scene - compensation
