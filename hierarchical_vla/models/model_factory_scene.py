"""
Factory for MambaVLAScene.
Baseline model_factory.py를 건드리지 않고 scene-aware variant 생성.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch

from MambaVLA.backbones.multi_img_obs_encoder import MultiImageObsEncoder
from MambaVLA.backbones.clip.clip_lang_encoder import LangClip
from MambaVLA.model_factory import create_mamba_backbone, OptimizerConfig, LRSchedulerConfig

from .vjepa_encoder import VJEPASceneEncoder
from .mambavla_scene import (
    SceneAwareMambaVLAPolicy,
    SceneAwareActionFlowMatching,
    MambaVLAScene,
)


def create_mambavla_scene_model(
    dataloader=None,
    camera_names: Optional[List[str]] = None,
    latent_dim: int = 256,
    action_dim: int = 7,
    lang_emb_dim: int = 512,
    scene_emb_dim: int = 256,           # VJEPASceneEncoder output_dim
    embed_dim: int = 256,
    obs_tok_len: Optional[int] = None,
    action_seq_len: int = 10,
    perception_seq_len: int = 1,
    state_dim: int = 7,
    device: str = "cuda",
    n_layer: int = 5,
    d_intermediate: int = 256,
    sampling_steps: int = 4,
    learning_rate: float = 1e-4,
    betas: Optional[List[float]] = None,
    clip_model_name: str = "ViT-B/32",
    # V-JEPA 관련
    vjepa_model_name: str = "facebook/vjepa2-vitg-fpc64-256",
    vjepa_n_finetune_layers: int = 4,
    vjepa_n_frames: int = 4,
):
    if camera_names is None:
        if dataloader is not None and hasattr(dataloader, "camera_names"):
            camera_names = dataloader.camera_names
        else:
            camera_names = ["agentview", "eye_in_hand"]

    if obs_tok_len is None:
        obs_tok_len = len(camera_names)

    if betas is None:
        betas = [0.9, 0.9]

    if dataloader is not None:
        if hasattr(dataloader, "action_dim"):
            action_dim = dataloader.action_dim
        if hasattr(dataloader, "state_dim"):
            state_dim = dataloader.state_dim

    # ── obs encoder (ResNet, baseline과 동일) ──────────────────────────────
    shape_meta = {
        "obs": {
            f"{cam}_image": {"shape": [3, 128, 128], "type": "rgb"}
            for cam in camera_names
        }
    }
    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model={
            "_target_": "MambaVLA.ResNetEncoder",
            "latent_dim": latent_dim,
            "pretrained": False,
            "freeze_backbone": False,
            "use_mlp": True,
        },
        resize_shape=None,
        crop_shape=None,
        random_crop=False,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
    ).to(device)

    # ── language encoder (CLIP, frozen) ───────────────────────────────────
    language_encoder = LangClip(
        freeze_backbone=True,
        model_name=clip_model_name,
    ).to(device)

    # ── scene encoder (V-JEPA, partially fine-tuned) ──────────────────────
    scene_encoder = VJEPASceneEncoder(
        model_name=vjepa_model_name,
        output_dim=scene_emb_dim,
        n_finetune_layers=vjepa_n_finetune_layers,
        n_frames=vjepa_n_frames,
    ).to(device)

    # ── Mamba backbone ─────────────────────────────────────────────────────
    mamba_encoder = create_mamba_backbone(
        embed_dim=embed_dim,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        device=device,
    )

    # ── SceneAwareMambaVLAPolicy ────────────────────────────────────────────
    policy = SceneAwareMambaVLAPolicy(
        encoder=mamba_encoder,
        latent_dim=latent_dim,
        action_dim=action_dim,
        lang_emb_dim=lang_emb_dim,
        scene_emb_dim=scene_emb_dim,
        device=device,
        embed_dim=embed_dim,
        embed_pdrob=0,
        lang_tok_len=1,
        obs_tok_len=obs_tok_len,
        action_seq_len=action_seq_len,
        linear_output=True,
        use_pos_emb=True,
    ).to(device)

    # ── Flow Matching ───────────────────────────────────────────────────────
    flow_model = SceneAwareActionFlowMatching(
        backbones=policy,
        ln=False,
        device=device,
    )

    # ── MambaVLAScene ───────────────────────────────────────────────────────
    optimizer_cfg = OptimizerConfig(
        transformer_weight_decay=0.05,
        obs_encoder_weight_decay=0.05,
        learning_rate=learning_rate,
        betas=betas,
    )
    lr_scheduler_cfg = LRSchedulerConfig()

    model = MambaVLAScene(
        flow_model=flow_model,
        obs_encoders=obs_encoder,
        language_encoders=language_encoder,
        scene_encoder=scene_encoder,
        optimizer_cfg=optimizer_cfg,
        lr_scheduler_cfg=lr_scheduler_cfg,
        action_dim=action_dim,
        perception_seq_len=perception_seq_len,
        action_seq_len=action_seq_len,
        cam_names=camera_names,
        device=device,
        state_dim=state_dim,
        latent_dim=latent_dim,
        sampling_steps=sampling_steps,
    )

    return model
