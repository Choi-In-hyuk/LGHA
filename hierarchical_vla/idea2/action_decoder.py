"""
Action Decoder (IDEA2 Stage 3).

codebook entry + z_scene → robot action sequence.
소량의 labeled robot data로 fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionDecoder(nn.Module):
    """
    (codebook_entry, z_scene) → action sequence

    codebook_entry: latent action "무엇을 할지" (256-dim)
    z_scene:        현재 scene state "어디서 할지" (256-dim)
    → 둘을 합쳐서 구체적인 robot joint commands로 디코딩
    """

    def __init__(
        self,
        codebook_dim: int = 256,
        scene_dim: int = 256,
        hidden_dim: int = 512,
        action_seq_len: int = 10,
        action_dim: int = 7,
    ):
        super().__init__()
        self.action_seq_len = action_seq_len
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(codebook_dim + scene_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_seq_len * action_dim),
        )

    def forward(self, codebook_entry: torch.Tensor, z_scene: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codebook_entry: (B, codebook_dim)
            z_scene:        (B, scene_dim)

        Returns:
            actions: (B, action_seq_len, action_dim)
        """
        x = torch.cat([codebook_entry, z_scene], dim=-1)
        out = self.net(x)
        return out.view(x.size(0), self.action_seq_len, self.action_dim)
