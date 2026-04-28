"""
Factory for MambaVLAWorld.
"""

from typing import List, Optional

from MambaVLA.backbones.multi_img_obs_encoder import MultiImageObsEncoder
from MambaVLA.model_factory import OptimizerConfig

from hierarchical_vla.models.vjepa_encoder import VJEPASceneEncoder
from .jepa_predictor import JEPAScenePredictor
from .world_policy import LatentWorldPolicy, LatentFlowMatching
from .mambavla_world import MambaVLAWorld


def create_mambavla_world_model(
    dataloader=None,
    camera_names: Optional[List[str]] = None,
    latent_dim: int = 256,
    action_dim: int = 7,
    lang_emb_dim: int = 512,
    scene_emb_dim: int = 256,
    embed_dim: int = 256,
    obs_tok_len: Optional[int] = None,
    action_seq_len: int = 10,
    perception_seq_len: int = 1,
    state_dim: int = 7,
    device: str = "cuda",
    n_heads: int = 4,
    n_policy_layers: int = 4,
    n_predictor_layers: int = 3,
    sampling_steps: int = 10,
    learning_rate: float = 1e-4,
    world_loss_weight: float = 1.0,
    # V-JEPA
    vjepa_model_name: str = "facebook/vjepa2-vitg-fpc64-256",
    vjepa_n_finetune_layers: int = 4,
    vjepa_n_frames: int = 4,
):
    if camera_names is None:
        camera_names = getattr(dataloader, "camera_names", ["agentview", "eye_in_hand"])
    if obs_tok_len is None:
        obs_tok_len = len(camera_names)
    if dataloader is not None:
        action_dim = getattr(dataloader, "action_dim", action_dim)
        state_dim  = getattr(dataloader, "state_dim",  state_dim)

    # ── Obs encoder (ResNet) ──────────────────────────────────────────────
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

    # ── Scene encoder (V-JEPA) ────────────────────────────────────────────
    scene_encoder = VJEPASceneEncoder(
        model_name=vjepa_model_name,
        output_dim=scene_emb_dim,
        n_finetune_layers=vjepa_n_finetune_layers,
        n_frames=vjepa_n_frames,
    ).to(device)

    # ── Scene predictor (z_t + lang → z_goal) ────────────────────────────
    predictor = JEPAScenePredictor(
        scene_dim=scene_emb_dim,
        lang_dim=lang_emb_dim,
        hidden_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_predictor_layers,
    ).to(device)

    # ── Latent policy (no Mamba) ──────────────────────────────────────────
    policy = LatentWorldPolicy(
        obs_dim=latent_dim,
        lang_dim=lang_emb_dim,
        scene_dim=scene_emb_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        action_seq_len=action_seq_len,
        n_heads=n_heads,
        n_layers=n_policy_layers,
    ).to(device)

    flow_model = LatentFlowMatching(policy=policy, device=device)

    # ── Optimizer config ──────────────────────────────────────────────────
    optimizer_cfg = OptimizerConfig(
        transformer_weight_decay=0.05,
        obs_encoder_weight_decay=0.05,
        learning_rate=learning_rate,
        betas=[0.9, 0.9],
    )

    model = MambaVLAWorld(
        obs_encoder=obs_encoder,
        scene_encoder=scene_encoder,
        predictor=predictor,
        flow_model=flow_model,
        optimizer_cfg=optimizer_cfg,
        action_dim=action_dim,
        action_seq_len=action_seq_len,
        perception_seq_len=perception_seq_len,
        cam_names=camera_names,
        device=device,
        latent_dim=latent_dim,
        state_dim=state_dim,
        sampling_steps=sampling_steps,
        world_loss_weight=world_loss_weight,
    )

    return model
