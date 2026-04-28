"""
JEPA Scene Predictor

(z_scene_current, lang_emb) → z_scene_goal

현재 scene latent와 언어 명령으로부터
primitive 실행 후의 목표 scene latent를 예측.

학습 target: 같은 demo의 마지막 프레임 latent (z_T)
"""

import torch
import torch.nn as nn


class JEPAScenePredictor(nn.Module):
    """
    Small transformer predictor.
    언어 명령이 가정하는 목표 scene 상태를 현재 scene latent에서 예측.

    Architecture:
        [z_scene, lang] → TransformerEncoder → z_scene 위치 출력 → z_goal
        z_goal = z_current + residual  (scene 변화량 예측)
    """

    def __init__(
        self,
        scene_dim: int = 256,
        lang_dim: int = 512,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
    ):
        super().__init__()

        self.scene_proj = nn.Linear(scene_dim, hidden_dim)
        self.lang_proj  = nn.Linear(lang_dim,  hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 예측: z_current에서 얼마나 변화할지 (residual delta)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, scene_dim),
        )

    def forward(self, z_scene: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_scene:  (B, scene_dim)
            lang_emb: (B, lang_dim) or (B, 1, lang_dim)

        Returns:
            z_goal: (B, scene_dim)  predicted goal scene latent
        """
        if lang_emb.dim() == 3:
            lang_emb = lang_emb.squeeze(1)  # (B, lang_dim)

        z_tok = self.scene_proj(z_scene).unsqueeze(1)  # (B, 1, hidden_dim)
        l_tok = self.lang_proj(lang_emb).unsqueeze(1)  # (B, 1, hidden_dim)

        # sequence: [z_scene_token, lang_token]
        seq = torch.cat([z_tok, l_tok], dim=1)         # (B, 2, hidden_dim)
        out = self.transformer(seq)                     # (B, 2, hidden_dim)

        # z_scene 위치(index 0)에서 delta 예측, residual 연결
        delta = self.delta_head(out[:, 0])              # (B, scene_dim)
        z_goal = z_scene + delta

        return z_goal
