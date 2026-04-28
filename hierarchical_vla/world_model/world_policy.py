"""
Latent World Policy

Mamba 토큰 시퀀스 없이 latent space에서 직접 액션 생성.

Conditioning:
  - z_obs  : ResNet visual latent (B, N, obs_dim)
  - z_lang : CLIP language latent (B, lang_dim)
  - z_diff : z_goal - z_current  (B, scene_dim)  ← 씬 변화량

Action generation:
  - Flow Matching (baseline과 동일)
  - 단, Mamba 대신 TransformerDecoder 사용
    (action tokens → context cross-attention)
"""

import torch
import torch.nn as nn

from MambaVLA.policy.policy import TimeEmbedding


class LatentWorldPolicy(nn.Module):
    """
    Action tokens가 latent context에 cross-attention하는 Flow Matching 정책.

    Token sequence 없음.
    Context: [sigma, lang, z_diff, obs_0, ..., obs_N]
    """

    def __init__(
        self,
        obs_dim: int = 256,
        lang_dim: int = 512,
        scene_dim: int = 256,
        action_dim: int = 7,
        embed_dim: int = 256,
        action_seq_len: int = 10,
        n_heads: int = 4,
        n_layers: int = 4,
    ):
        super().__init__()

        self.action_seq_len = action_seq_len
        self.action_dim = action_dim

        # input projections
        self.obs_proj   = nn.Linear(obs_dim,   embed_dim)
        self.lang_proj  = nn.Linear(lang_dim,  embed_dim)
        self.diff_proj  = nn.Linear(scene_dim, embed_dim)
        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.sigma_emb  = TimeEmbedding(embed_dim)

        # action tokens cross-attend to context
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.action_head = nn.Linear(embed_dim, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        z_obs: torch.Tensor,    # (B, N, obs_dim)
        lang_emb: torch.Tensor, # (B, lang_dim) or (B, 1, lang_dim)
        z_diff: torch.Tensor,   # (B, scene_dim)
        actions: torch.Tensor,  # (B, M, action_dim)
        sigma: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        """Returns predicted velocity field: (B, M, action_dim)"""
        if lang_emb.dim() == 3:
            lang_emb = lang_emb.squeeze(1)  # (B, lang_dim)

        # context tokens: [sigma(1), lang(1), z_diff(1), obs(N)]
        sigma_tok = self.sigma_emb(sigma)                    # (B, 1, embed_dim)
        lang_tok  = self.lang_proj(lang_emb).unsqueeze(1)   # (B, 1, embed_dim)
        diff_tok  = self.diff_proj(z_diff).unsqueeze(1)     # (B, 1, embed_dim)
        obs_tok   = self.obs_proj(z_obs)                    # (B, N, embed_dim)

        context = torch.cat([sigma_tok, lang_tok, diff_tok, obs_tok], dim=1)

        # action tokens attend to context
        act_tok = self.action_emb(actions)                  # (B, M, embed_dim)
        out = self.decoder(act_tok, context)                # (B, M, embed_dim)

        return self.action_head(out)                        # (B, M, action_dim)


class LatentFlowMatching(nn.Module):
    """Flow Matching wrapper for LatentWorldPolicy."""

    def __init__(self, policy: LatentWorldPolicy, device: str = "cpu"):
        super().__init__()
        self.model = policy.to(device)

    def forward(self, actions, z_obs, lang_emb, z_diff):
        batch_size = actions.size(0)
        time_steps = torch.rand(batch_size, device=actions.device)
        time_exp   = time_steps.view(batch_size, *([1] * len(actions.shape[1:])))

        noise_actions        = torch.randn_like(actions)
        interpolated_actions = (1 - time_exp) * actions + time_exp * noise_actions

        velocity_pred = self.model(z_obs, lang_emb, z_diff, interpolated_actions, time_steps)

        batchwise_mse = ((noise_actions - actions - velocity_pred) ** 2).mean(
            dim=list(range(1, len(actions.shape)))
        )
        return batchwise_mse.mean()

    @torch.no_grad()
    def generate_actions(self, noise_actions, z_obs, lang_emb, z_diff, sample_steps: int = 10):
        batch_size = noise_actions.size(0)
        step_size  = torch.tensor(
            [1.0 / sample_steps] * batch_size, device=noise_actions.device
        ).view(batch_size, *([1] * len(noise_actions.shape[1:])))

        for i in range(sample_steps, 0, -1):
            time_step = torch.tensor(
                [i / sample_steps] * batch_size, device=noise_actions.device
            )
            velocity_pred = self.model(z_obs, lang_emb, z_diff, noise_actions, time_step)
            noise_actions = noise_actions - step_size * velocity_pred

        return noise_actions
