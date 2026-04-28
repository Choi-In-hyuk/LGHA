"""
Latent VLA Policy (IDEA2 v2).

V-JEPA 완전 제거. SceneEncoder(frozen ResNet-18)로 z_scene 일관 계산.
EpisodeMemory의 delta_history → ContextEncoder → context_vec 를 ActionDecoder에 주입.

Token sequence (Mamba): [obs_tokens..., lang_token]
ActionDecoder 입력:     codebook[k] + (z_scene + context_vec)
"""

import logging
from collections import deque
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class LatentVLAPolicy(nn.Module):
    """
    Mamba backbone + codebook index classification head.
    입력: obs_tokens (ResNet) + lang embedding
    출력: codebook index logits (B, K)
    """

    def __init__(
        self,
        encoder: Any,           # Mamba backbone
        latent_dim: int,
        lang_emb_dim: int,
        K: int,
        embed_dim: int,
        obs_tok_len: int,
        embed_pdrop: float = 0.0,
        use_pos_emb: bool = True,
    ):
        super().__init__()

        self.obs_tok_len = obs_tok_len
        self.seq_size = obs_tok_len + 1  # obs + lang (lang last)

        self.tok_emb  = nn.Linear(latent_dim,   embed_dim)
        self.lang_emb = nn.Linear(lang_emb_dim, embed_dim)

        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))

        self.drop = nn.Dropout(embed_pdrop)
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, K)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, states: torch.Tensor, lang_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states:    (B, T, latent_dim)  obs tokens
            lang_cond: (B, 1, lang_emb_dim)

        Returns:
            logits: (B, K)
        """
        state_x = self.tok_emb(states)
        if self.use_pos_emb:
            state_x = state_x + self.pos_emb[:, :states.size(1)]
        state_x = self.drop(state_x)

        lang_x = self.lang_emb(lang_cond)
        if self.use_pos_emb:
            lang_x = lang_x + self.pos_emb[:, states.size(1):states.size(1) + 1]
        lang_x = self.drop(lang_x)

        # obs 먼저, lang 마지막 → Mamba causal: lang이 모든 obs 정보 수집
        seq = torch.cat([state_x, lang_x], dim=1)
        out = self.encoder(seq)

        return self.classifier(out[:, -1])  # (B, K)


class IDEA2Model(nn.Module):
    """
    IDEA2 v2 전체 모델.

    V-JEPA 제거. SceneEncoder(frozen ResNet-18)로 z_scene 일관 계산.
    EpisodeMemory + ContextEncoder → context_vec → ActionDecoder 주입.

    Stage 2: (obs_tokens, lang) → codebook_k  (CrossEntropy)
    Stage 3: codebook[k] + (z_scene + context_vec) → actions  (MSE)
    Inference: predict(obs_dict, context_vec) → (actions, z_scene, k)
    """

    def __init__(
        self,
        policy: LatentVLAPolicy,
        obs_encoder: Any,           # MultiImageObsEncoder (trainable)
        scene_encoder: Any,         # SceneEncoder (frozen ResNet-18)
        vqvae: Any,                 # VQVAEModel (frozen after Stage 1)
        action_decoder: Any,        # ActionDecoder
        action_seq_len: int,
        action_dim: int,
        perception_seq_len: int,
        cam_names: list,
        device: str = "cuda",
        success_threshold: float = 2.0,  # latent distance 성공 판정 threshold
    ):
        super().__init__()

        self.device = device
        self.scaler = None
        self.success_threshold = success_threshold

        self.obs_encoder   = obs_encoder.to(device)
        self.scene_encoder = scene_encoder.to(device)
        self.policy        = policy.to(device)
        self.vqvae         = vqvae.to(device)
        self.action_decoder = action_decoder.to(device)

        self.action_seq_len     = action_seq_len
        self.action_dim         = action_dim
        self.perception_seq_len = perception_seq_len
        self.cam_names          = cam_names

        # rollout 상태
        self.rollout_step_counter = 0
        self.obs_seq: dict[str, deque] = {}
        self.pred_action_seq: torch.Tensor = None
        self.last_k: int = -1

    def set_scaler(self, scaler):
        self.scaler = scaler

    def reset(self):
        self.rollout_step_counter = 0
        self.obs_seq = {}
        self.pred_action_seq = None
        self.last_k = -1

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _prepare_obs(self, obs_dict: dict) -> dict:
        """(B, T, C, H, W) → (B*T, C, H, W) for img_encoder."""
        B, T, C, H, W = obs_dict[f"{self.cam_names[0]}_image"].shape
        prepared = dict(obs_dict)
        for cam in self.cam_names:
            prepared[f"{cam}_image"] = obs_dict[f"{cam}_image"].view(B * T, C, H, W)
        return prepared, B, T

    def _encode_obs_tokens(self, obs_dict: dict) -> tuple:
        """
        이미지 → obs_tokens (Mamba 입력용)
        Returns:
            obs_tokens: (B*T, obs_tok_len, latent_dim)
            lang_embed: (B, 1, lang_emb_dim)
        """
        lang_embed = obs_dict["lang_emb"]
        prepared, B, T = self._prepare_obs(obs_dict)
        obs_tokens = self.obs_encoder(prepared)
        return obs_tokens, lang_embed

    def get_z_scene(self, obs_dict: dict) -> torch.Tensor:
        """
        현재 agentview 이미지 → z_scene.
        inference 루프에서 z_before / z_after 계산에 사용.

        Args:
            obs_dict: {"agentview_image": (B, 1, C, H, W)} 또는 (B, C, H, W)

        Returns:
            z_scene: (B, scene_dim)
        """
        img = obs_dict[f"{self.cam_names[0]}_image"]
        if img.dim() == 5:
            # (B, T, C, H, W) → 마지막 프레임
            img = img[:, -1]
        with torch.no_grad():
            return self.scene_encoder(img.to(self.device))

    # ── Training forwards ────────────────────────────────────────────────────

    def forward_stage2(
        self,
        obs_dict: dict,
        target_indices: torch.Tensor,
        context_vec: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Stage 2: (obs_tokens, lang) → codebook index (CrossEntropy).
        context_vec: (B, scene_dim) — None이면 zero (history 없는 상황)
        """
        obs_tokens, lang_embed = self._encode_obs_tokens(obs_dict)

        if lang_embed.dim() == 2:
            lang_embed = lang_embed.unsqueeze(1)

        logits = self.policy(obs_tokens, lang_embed)  # (B, K)
        return F.cross_entropy(logits, target_indices)

    def forward_stage3(
        self,
        obs_dict: dict,
        actions: torch.Tensor,
        target_indices: torch.Tensor,
        context_vec: torch.Tensor = None,
        z_scene: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Stage 3: codebook[k] + z_compensated → actions (MSE).
        z_compensated = z_scene - cumulative_delta
        → 바뀐 환경을 학습 분포로 보정.

        z_scene: 캐시에서 직접 넘기면 V-JEPA live forward 생략 → 빠른 학습.
                 None이면 scene_encoder로 live 계산 (inference 시).
        """
        if z_scene is None:
            img = obs_dict[f"{self.cam_names[0]}_image"]
            if img.dim() == 5:
                img = img[:, -1]
            z_scene = self.scene_encoder(img)

        # 누적 delta를 빼서 학습 분포로 보정
        if context_vec is None:
            context_vec = torch.zeros_like(z_scene)
        z_input = z_scene - context_vec  # (B, scene_dim)

        codebook_entry = self.vqvae.vq.lookup(target_indices)   # (B, D)
        pred_actions   = self.action_decoder(codebook_entry, z_input)  # (B, T, 7)
        return F.mse_loss(pred_actions, actions)

    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        obs_dict: dict,
        context_vec: torch.Tensor = None,
    ) -> tuple:
        """
        추론: (obs, context) → (action, z_scene, codebook_k).

        Args:
            obs_dict:    단일 타임스텝 observation
            context_vec: (scene_dim,) or (1, scene_dim) — None이면 zero

        Returns:
            action:  (action_dim,) 현재 스텝 action
            z_scene: (1, scene_dim) 현재 z_scene (delta 계산용)
            k:       int codebook index
        """
        # ── Observation history 구축 ────────────────────────────────────────
        if not self.obs_seq:
            for key in obs_dict:
                self.obs_seq[key] = deque(maxlen=self.perception_seq_len)

        for key in obs_dict:
            self.obs_seq[key].append(obs_dict[key])

        obs_buffered = {}
        for key in obs_dict:
            if key == "lang_emb":
                obs_buffered[key] = obs_dict[key]
                continue
            stacked = torch.cat(list(self.obs_seq[key]), dim=1)  # (1, t, ...)
            if stacked.shape[1] < self.perception_seq_len:
                pad = einops.repeat(
                    stacked[:, 0], "b ... -> b t ...",
                    t=self.perception_seq_len - stacked.shape[1],
                )
                stacked = torch.cat([pad, stacked], dim=1)
            obs_buffered[key] = stacked

        # ── 첫 번째 스텝에서만 action sequence 계산 ──────────────────────────
        if self.rollout_step_counter == 0:
            self.eval()

            obs_tokens, lang_embed = self._encode_obs_tokens(obs_buffered)
            if lang_embed.dim() == 2:
                lang_embed = lang_embed.unsqueeze(1)

            logits = self.policy(obs_tokens, lang_embed)  # (1, K)
            k      = logits.argmax(dim=-1)                # (1,)
            self.last_k = k[0].item()

            # z_scene
            img = obs_buffered[f"{self.cam_names[0]}_image"][:, -1]
            z_scene = self.scene_encoder(img)             # (1, scene_dim)

            # context
            if context_vec is None:
                ctx = torch.zeros_like(z_scene)
            else:
                ctx = context_vec.to(self.device)
                if ctx.dim() == 1:
                    ctx = ctx.unsqueeze(0)
            # 누적 delta를 빼서 학습 분포로 보정
            z_input = z_scene - ctx                       # (1, scene_dim)

            codebook_entry = self.vqvae.vq.lookup(k)     # (1, D)
            pred_actions   = self.action_decoder(codebook_entry, z_input)  # (1, T, 7)

            _log = logging.getLogger("idea2.predict")
            _log.info(
                f"[predict] k={self.last_k} | z_scene_norm={z_scene.norm().item():.3f} "
                f"| ctx_norm={ctx.norm().item():.3f} "
                f"| pred_actions max={pred_actions.abs().max().item():.3f}"
            )

            if self.scaler is not None:
                pred_actions = self.scaler.inverse_scale_output(pred_actions)
                _log.info(f"[predict] after scaler max={pred_actions.abs().max().item():.3f}")

            self.pred_action_seq = pred_actions
            self._last_z_scene   = z_scene
            self._last_k         = k[0].item()

        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        self.rollout_step_counter += 1
        if self.rollout_step_counter >= self.action_seq_len:
            self.rollout_step_counter = 0

        return current_action, self._last_z_scene, self._last_k

    def check_success_latent(self, delta_actual: torch.Tensor) -> bool:
        """
        실제 delta_z와 codebook entry의 거리로 성공 판정.
        reward가 없는 중간 단계에서 보조 판단에 사용.
        """
        if self.last_k < 0:
            return False
        expected = self.vqvae.vq.lookup(
            torch.tensor([self.last_k], device=self.device)
        ).squeeze(0)
        dist = (delta_actual.to(self.device) - expected).norm().item()
        return dist < self.success_threshold

    def configure_optimizer(self, lr: float = 1e-4, weight_decay: float = 0.05):
        """
        Stage 2/3 학습용 optimizer.
        scene_encoder(frozen) + vqvae(frozen)는 제외.
        """
        frozen_prefixes = ("scene_encoder.", "vqvae.")
        params = [
            p for name, p in self.named_parameters()
            if p.requires_grad and not any(name.startswith(fp) for fp in frozen_prefixes)
        ]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
