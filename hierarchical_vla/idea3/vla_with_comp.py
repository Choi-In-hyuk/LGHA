"""
MambaVLAWithComp: MambaVLA + compensation token.

comp_delta (환경 변화량)를 Linear로 embed_dim으로 projection해서
obs_embed에 additive bias로 주입.

학습 목표: comp_delta가 있을 때도 올바른 action 예측.
추론 시: EpisodeMemory의 누적 delta를 comp_delta로 제공.
"""

import torch
import torch.nn as nn

from MambaVLA.mambavla_model import MambaVLA


class MambaVLAWithComp(MambaVLA):
    """
    MambaVLA에 compensation projection을 추가한 서브클래스.

    obs_embed: (B, num_cams, latent_dim)
    comp_delta: (B, num_cams, latent_dim)  — 학습 시 batch
               (num_cams, latent_dim)       — 추론 시 단일

    _input_embeddings에서:
        comp_bias = comp_proj(comp_delta.flatten()) → (latent_dim,)
        obs_embed = obs_embed + comp_bias            (broadcast)
    """

    NUM_CAMS  = 2
    LATENT_DIM = 256

    def __init__(self, *args, embed_dim: int = 256, **kwargs):
        super().__init__(*args, **kwargs)

        comp_input_dim = self.NUM_CAMS * self.LATENT_DIM  # 512

        self.comp_norm = nn.LayerNorm(comp_input_dim)
        self.comp_proj = nn.Linear(comp_input_dim, self.LATENT_DIM)

        # zero init → fine-tuning 초기엔 pretrained 동작 그대로
        nn.init.zeros_(self.comp_proj.weight)
        nn.init.zeros_(self.comp_proj.bias)

        # 추론용 single delta
        self._comp_delta = None

    # ── 추론 시 사용 ─────────────────────────────────────────────────────────

    def set_comp_delta(self, delta: torch.Tensor | None):
        """
        추론 시 호출.
        delta: (num_cams, latent_dim) or None
        """
        self._comp_delta = delta

    # ── core override ────────────────────────────────────────────────────────

    def _input_embeddings(self, obs_dict: dict):
        obs_embed, lang_embed = super()._input_embeddings(obs_dict)
        # obs_embed: (B, num_cams, latent_dim)

        comp_delta = obs_dict.get("comp_delta", None)  # 학습 시 batch

        if comp_delta is None:
            comp_delta = self._comp_delta  # 추론 시 single

        if comp_delta is not None:
            B = obs_embed.shape[0]
            delta = comp_delta.to(obs_embed.device).float()

            # batch or single → (B, num_cams, latent_dim)
            if delta.dim() == 2:                    # (num_cams, latent_dim) 추론
                delta = delta.unsqueeze(0).expand(B, -1, -1)

            delta_flat = delta.flatten(1)           # (B, num_cams*latent_dim)
            delta_flat = self.comp_norm(delta_flat) # LayerNorm
            comp_bias  = self.comp_proj(delta_flat) # (B, latent_dim)
            obs_embed  = obs_embed + comp_bias.unsqueeze(1)  # broadcast over cams

        return obs_embed, lang_embed
