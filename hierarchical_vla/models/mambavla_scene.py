"""
Scene-Aware MambaVLA

Baseline MambaVLA에 V-JEPA scene token을 추가한 variant.
baseline 코드(/home/choi/LGHA/MambaVLA/)는 수정하지 않음.

Token sequence 변경:
  Before: [sigma, lang(1), obs(N), action(M)]
  After:  [sigma, lang(1), scene(1), obs(N), action(M)]
"""

import os
import pickle
import logging
from collections import deque
from typing import Any, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from MambaVLA.policy.policy import TimeEmbedding
from MambaVLA.policy.flowmatching import ActionFLowMatching

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Scene-aware Policy (MambaVLAPolicy + scene_token)
# ---------------------------------------------------------------------------

class SceneAwareMambaVLAPolicy(nn.Module):
    """
    MambaVLAPolicy에 scene_token 1개 추가.
    token sequence: [sigma, lang, scene, obs_0..obs_N, action_0..action_M]
    """

    def __init__(
        self,
        encoder: Any,
        latent_dim: int,
        action_dim: int,
        lang_emb_dim: int,
        scene_emb_dim: int,          # V-JEPA output_dim (e.g. 256)
        device: str,
        embed_dim: int,
        embed_pdrob: float,
        lang_tok_len: int,
        obs_tok_len: int,
        action_seq_len: int,
        linear_output: bool = False,
        use_pos_emb: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        self.device = device

        self.lang_tok_len = lang_tok_len
        self.scene_tok_len = 1
        self.obs_tok_len = obs_tok_len
        self.action_seq_len = action_seq_len

        # sigma + lang + scene + obs + action
        self.seq_size = lang_tok_len + self.scene_tok_len + obs_tok_len + action_seq_len

        # token projections
        self.tok_emb    = nn.Linear(latent_dim,    embed_dim)   # obs
        self.lang_emb   = nn.Linear(lang_emb_dim,  embed_dim)   # language
        self.scene_emb  = nn.Linear(scene_emb_dim, embed_dim)   # scene  ← NEW
        self.action_emb = nn.Linear(action_dim,    embed_dim)   # action

        self.sigma_emb = TimeEmbedding(embed_dim)

        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))

        self.drop = nn.Dropout(embed_pdrob)
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, action_dim),
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, states, actions, lang_cond, scene_cond, sigma):
        """
        Args:
            states:     (B, T, latent_dim)
            actions:    (B, M, action_dim)
            lang_cond:  (B, lang_tok_len, lang_emb_dim)
            scene_cond: (B, 1, scene_emb_dim)   ← NEW
            sigma:      (B,)
        """
        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, _ = states.size()
        _, t_a, _ = actions.size()
        offset = 0

        # language token
        lang_embed = self.lang_emb(lang_cond)
        if self.use_pos_emb:
            lang_embed = lang_embed + self.pos_emb[:, offset:offset + self.lang_tok_len]
        lang_x = self.drop(lang_embed)
        offset += self.lang_tok_len

        # scene token  ← NEW
        scene_embed = self.scene_emb(scene_cond)   # (B, 1, embed_dim)
        if self.use_pos_emb:
            scene_embed = scene_embed + self.pos_emb[:, offset:offset + self.scene_tok_len]
        scene_x = self.drop(scene_embed)
        offset += self.scene_tok_len

        # obs tokens
        state_embed = self.tok_emb(states)
        if self.use_pos_emb:
            state_embed = state_embed + self.pos_emb[:, offset:offset + t]
        state_x = self.drop(state_embed)
        offset += t

        # action tokens
        action_embed = self.action_emb(actions)
        if self.use_pos_emb:
            action_embed = action_embed + self.pos_emb[:, offset:offset + t_a]
        action_x = self.drop(action_embed)

        # sigma embedding (prepended, not in pos_emb)
        emb_t = self.sigma_emb(sigma)

        # full sequence: [sigma, lang, scene, obs, action]
        input_seq = torch.cat([emb_t, lang_x, scene_x, state_x, action_x], dim=1)

        encoder_output = self.encoder(input_seq)
        pred_actions = self.action_pred(encoder_output[:, -self.action_seq_len:, :])

        return pred_actions


# ---------------------------------------------------------------------------
# 2. Scene-aware Flow Matching (scene_cond을 policy에 전달)
# ---------------------------------------------------------------------------

class SceneAwareActionFlowMatching(nn.Module):
    def __init__(self, backbones: Any, ln: bool = False, device: str = "cpu"):
        super().__init__()
        self.model = backbones.to(device)
        self.ln = ln

    def forward(self, actions, state, lang_embed, scene_embed):
        batch_size = actions.size(0)
        if self.ln:
            noise_t = torch.randn((batch_size,)).to(actions.device)
            time_steps = torch.sigmoid(noise_t)
        else:
            time_steps = torch.rand((batch_size,)).to(actions.device)

        time_expanded = time_steps.view([batch_size, *([1] * len(actions.shape[1:]))])
        noise_actions = torch.randn_like(actions)
        interpolated_actions = (1 - time_expanded) * actions + time_expanded * noise_actions

        velocity_pred = self.model(
            states=state,
            actions=interpolated_actions,
            lang_cond=lang_embed,
            scene_cond=scene_embed,
            sigma=time_steps,
        )

        batchwise_mse = ((noise_actions - actions - velocity_pred) ** 2).mean(
            dim=list(range(1, len(actions.shape)))
        )
        time_loss_pairs = list(zip(time_steps.tolist(), batchwise_mse.detach().cpu().tolist()))
        return batchwise_mse.mean(), time_loss_pairs

    @torch.no_grad()
    def generate_actions(self, noise_actions, state, lang_embed, scene_embed, sample_steps=50):
        batch_size = noise_actions.size(0)
        step_size = torch.tensor(
            [1.0 / sample_steps] * batch_size,
            device=noise_actions.device,
        ).view([batch_size, *([1] * len(noise_actions.shape[1:]))])

        for i in range(sample_steps, 0, -1):
            time_step = torch.tensor(
                [i / sample_steps] * batch_size, device=noise_actions.device
            )
            velocity_pred = self.model(
                states=state,
                actions=noise_actions,
                lang_cond=lang_embed,
                scene_cond=scene_embed,
                sigma=time_step,
            )
            noise_actions = noise_actions - step_size * velocity_pred

        return noise_actions


# ---------------------------------------------------------------------------
# 3. MambaVLAScene (top-level model)
# ---------------------------------------------------------------------------

class MambaVLAScene(nn.Module):
    """
    Baseline MambaVLA + V-JEPA scene encoder.

    추가 입력: obs_dict["agentview_image"] → VJEPASceneEncoder → scene_token
    """

    def __init__(
        self,
        flow_model: SceneAwareActionFlowMatching,
        obs_encoders: Any,
        language_encoders: Any,
        scene_encoder: Any,          # VJEPASceneEncoder
        optimizer_cfg: Any,
        lr_scheduler_cfg: Any,
        action_dim: int,
        perception_seq_len: int,
        action_seq_len: int,
        cam_names: list,
        device: str = "cuda",
        state_dim: int = 7,
        latent_dim: int = 256,
        sampling_steps: int = 4,
    ):
        super().__init__()

        self.device = device
        self.working_dir = os.getcwd()
        self.scaler = None

        self.img_encoder      = obs_encoders.to(device)
        self.language_encoder = language_encoders.to(device)
        self.scene_encoder    = scene_encoder.to(device)   # ← NEW
        self.model            = flow_model

        self.state_emb = nn.Linear(state_dim, latent_dim).to(device)
        self.cam_names = cam_names

        self.rollout_step_counter = 0
        self.action_seq_len   = action_seq_len
        self.perception_seq_len = perception_seq_len
        self.action_dim = action_dim

        self.obs_seq: dict[str, deque] = {}

        self.optimizer_cfg    = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.sampling_steps   = sampling_steps

    def set_scaler(self, scaler):
        self.scaler = scaler

    def reset(self):
        self.rollout_step_counter = 0
        self.obs_seq = {}

    def _input_embeddings(self, obs_dict):
        """
        Returns:
            obs_embed:   (B, obs_tok_len, latent_dim)
            lang_embed:  (B, 1, lang_emb_dim)
            scene_embed: (B, 1, scene_emb_dim)   ← NEW
        """
        # language
        lang_embed = obs_dict["lang_emb"]

        # visual (ResNet)
        B, T, C, H, W = obs_dict[f"{self.cam_names[0]}_image"].shape
        for cam in self.cam_names:
            obs_dict[f"{cam}_image"] = obs_dict[f"{cam}_image"].view(B * T, C, H, W)

        obs_embed = self.img_encoder(obs_dict)  # (B*T, n_cameras, latent_dim)

        # scene (V-JEPA) — 캐시 있으면 바로 사용, 없으면 encoder 실행
        if "z_scene_cache" in obs_dict:
            scene_embed = obs_dict["z_scene_cache"]        # (B, 1, scene_dim)
        else:
            agentview_flat = obs_dict["agentview_image"]   # (B*T, C, H, W)
            current_frame  = agentview_flat.view(B, T, C, H, W)[:, -1]
            z_scene        = self.scene_encoder(current_frame)
            scene_embed    = z_scene.unsqueeze(1)          # (B, 1, scene_dim)

        return obs_embed, lang_embed, scene_embed

    def forward(self, obs_dict, actions=None):
        obs_embed, lang_embed, scene_embed = self._input_embeddings(obs_dict)

        if self.training and actions is not None:
            loss, _ = self.model(actions, obs_embed, lang_embed, scene_embed)
            return loss

        noise_actions = torch.randn(
            (len(obs_embed), self.action_seq_len, self.action_dim), device=self.device
        )
        pred_act_seq = self.model.generate_actions(
            noise_actions, obs_embed, lang_embed, scene_embed,
            sample_steps=self.sampling_steps,
        )
        return pred_act_seq

    @torch.no_grad()
    def predict(self, obs_dict: dict) -> torch.Tensor:
        if not self.obs_seq:
            for key in obs_dict:
                self.obs_seq[key] = deque(maxlen=self.perception_seq_len)

        for key in obs_dict:
            self.obs_seq[key].append(obs_dict[key])
            if key == "lang_emb":
                continue
            obs_dict[key] = torch.cat(list(self.obs_seq[key]), dim=1)
            if obs_dict[key].shape[1] < self.perception_seq_len:
                pad = einops.repeat(
                    obs_dict[key][:, 0], "b ... -> b t ...",
                    t=self.perception_seq_len - obs_dict[key].shape[1],
                )
                obs_dict[key] = torch.cat([pad, obs_dict[key]], dim=1)

        if self.rollout_step_counter == 0:
            self.eval()
            pred_action_seq = self(obs_dict)[:, :self.action_seq_len]
            pred_action_seq = self.scaler.inverse_scale_output(pred_action_seq)
            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.action_seq_len:
            self.rollout_step_counter = 0

        return current_action

    def configure_optimizer(self):
        """학습률 그룹: scene_encoder(낮음) / 나머지(높음)."""
        lr = self.optimizer_cfg.learning_rate
        wd_tf = self.optimizer_cfg.transformer_weight_decay
        wd_obs = self.optimizer_cfg.obs_encoder_weight_decay

        scene_params   = list(self.scene_encoder.parameters())
        scene_ids      = {id(p) for p in scene_params}
        other_params   = [p for p in self.parameters() if id(p) not in scene_ids]

        param_groups = [
            {"params": [p for p in scene_params if p.requires_grad],
             "lr": lr * 0.1,   # scene encoder는 낮은 학습률
             "weight_decay": wd_obs},
            {"params": other_params,
             "lr": lr,
             "weight_decay": wd_tf},
        ]
        return torch.optim.AdamW(param_groups)
