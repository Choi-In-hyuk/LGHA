"""
MambaVLAWorld

JEPA world model + latent policy.

학습 2단계:
  1. World model: predictor(z_t, lang) → z_T (demo 마지막 프레임 latent)
  2. Policy:      (z_obs, lang, z_T - z_t) → action  (GT goal으로 teacher forcing)

추론:
  z_goal = predictor(z_current, lang)   ← 예측된 목표
  action = policy(z_obs, lang, z_goal - z_current)

토큰 시퀀스 없음. 순수 latent space 연산.
"""

import os
import pickle
import logging
from collections import deque
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class MambaVLAWorld(nn.Module):
    """
    Top-level world model.

    Components:
        obs_encoder   : MultiImageObsEncoder (ResNet)
        scene_encoder : VJEPASceneEncoder
        predictor     : JEPAScenePredictor  (z_t + lang → z_goal)
        flow_model    : LatentFlowMatching  (latent policy)
    """

    def __init__(
        self,
        obs_encoder: Any,
        scene_encoder: Any,
        predictor: Any,
        flow_model: Any,
        optimizer_cfg: Any,
        action_dim: int = 7,
        action_seq_len: int = 10,
        perception_seq_len: int = 1,
        cam_names: list = None,
        device: str = "cuda",
        latent_dim: int = 256,
        state_dim: int = 7,
        sampling_steps: int = 10,
        world_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.device = device
        self.scaler = None

        self.img_encoder   = obs_encoder.to(device)
        self.scene_encoder = scene_encoder.to(device)
        self.predictor     = predictor.to(device)
        self.flow_model    = flow_model

        self.state_emb = nn.Linear(state_dim, latent_dim).to(device)
        self.cam_names = cam_names or ["agentview", "eye_in_hand"]

        self.action_dim         = action_dim
        self.action_seq_len     = action_seq_len
        self.perception_seq_len = perception_seq_len
        self.sampling_steps     = sampling_steps
        self.world_loss_weight  = world_loss_weight

        self.rollout_step_counter = 0
        self.obs_seq: dict = {}

        self.optimizer_cfg = optimizer_cfg

    def set_scaler(self, scaler):
        self.scaler = scaler

    def reset(self):
        self.rollout_step_counter = 0
        self.obs_seq = {}

    def _encode_obs(self, obs_dict):
        """
        Returns:
            z_obs:     (B, T, latent_dim)   ResNet visual latent
            lang_emb:  (B, 1, lang_dim)     CLIP language embedding
            z_current: (B, scene_dim)       V-JEPA current scene latent
        """
        lang_emb = obs_dict["lang_emb"]
        if lang_emb.dim() == 2:
            lang_emb = lang_emb.unsqueeze(1)

        # ResNet obs encoding
        B, T, C, H, W = obs_dict[f"{self.cam_names[0]}_image"].shape
        for cam in self.cam_names:
            obs_dict[f"{cam}_image"] = obs_dict[f"{cam}_image"].view(B * T, C, H, W)

        z_obs = self.img_encoder(obs_dict, lang_emb)
        z_obs = z_obs.view(B, T, -1)

        # V-JEPA current scene
        agentview_flat = obs_dict["agentview_image"]          # (B*T, C, H, W)
        current_frame  = agentview_flat.view(B, T, C, H, W)[:, -1]  # (B, C, H, W)
        z_current      = self.scene_encoder(current_frame)    # (B, scene_dim)

        return z_obs, lang_emb, z_current

    def forward(self, obs_dict, actions=None, goal_frame=None):
        """
        Training:
            obs_dict:   current observation
            actions:    (B, M, action_dim)
            goal_frame: (B, C, H, W) — demo 마지막 프레임 (goal GT)

            Returns total_loss, world_loss, policy_loss

        Inference (actions=None):
            Returns predicted action sequence
        """
        z_obs, lang_emb, z_current = self._encode_obs(obs_dict)

        if self.training and actions is not None:
            # ── World model loss ──────────────────────────────────────────
            # GT goal: JEPA encode of the last demo frame
            with torch.no_grad():
                z_goal_gt = self.scene_encoder(goal_frame)  # (B, scene_dim)

            z_goal_pred = self.predictor(z_current, lang_emb)
            world_loss  = F.mse_loss(z_goal_pred, z_goal_gt)

            # ── Policy loss (teacher forcing with GT goal) ────────────────
            z_diff      = z_goal_gt - z_current             # (B, scene_dim)
            policy_loss = self.flow_model(actions, z_obs, lang_emb, z_diff)

            total_loss  = policy_loss + self.world_loss_weight * world_loss
            return total_loss, world_loss, policy_loss

        # ── Inference ─────────────────────────────────────────────────────
        z_goal_pred = self.predictor(z_current, lang_emb)
        z_diff      = z_goal_pred - z_current

        noise_actions = torch.randn(
            (z_obs.size(0), self.action_seq_len, self.action_dim), device=self.device
        )
        pred_actions = self.flow_model.generate_actions(
            noise_actions, z_obs, lang_emb, z_diff,
            sample_steps=self.sampling_steps,
        )
        return pred_actions

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
        """
        param groups:
          - scene_encoder + predictor: 낮은 LR (JEPA fine-tune)
          - policy (obs_encoder + flow_model): 기본 LR
        """
        lr   = self.optimizer_cfg.learning_rate
        wd   = self.optimizer_cfg.transformer_weight_decay

        jepa_params  = list(self.scene_encoder.parameters()) + list(self.predictor.parameters())
        jepa_ids     = {id(p) for p in jepa_params}
        other_params = [p for p in self.parameters() if id(p) not in jepa_ids]

        return torch.optim.AdamW([
            {"params": [p for p in jepa_params  if p.requires_grad], "lr": lr * 0.1, "weight_decay": wd},
            {"params": [p for p in other_params if p.requires_grad], "lr": lr,       "weight_decay": wd},
        ])
