"""
IDEA2 v2 Model Factory.

V-JEPA 제거. SceneEncoder(frozen ResNet-18) 사용.
precompute_resnet.py 가 저장한 scene_encoder.pt 를 로드해 일관성 보장.
"""

import logging
import os

import torch

from MambaVLA.backbones.multi_img_obs_encoder import MultiImageObsEncoder
from MambaVLA.model_factory import create_mamba_backbone

from .scene_encoder import SceneEncoder
from .context_encoder import ContextEncoder
from .vqvae import VQVAEModel
from .action_decoder import ActionDecoder
from .latent_vla import LatentVLAPolicy, IDEA2Model

log = logging.getLogger(__name__)


def create_idea2_model(
    camera_names: list = None,
    latent_dim: int = 256,
    action_dim: int = 7,
    lang_emb_dim: int = 512,
    scene_emb_dim: int = 256,
    embed_dim: int = 256,
    obs_tok_len: int = None,
    action_seq_len: int = 10,
    perception_seq_len: int = 1,
    n_layer: int = 5,
    d_intermediate: int = 256,
    K: int = 64,
    device: str = "cuda",
    vqvae_path: str = None,          # Stage 1 학습된 VQ-VAE 경로
    scene_encoder_path: str = None,  # precompute_scene.py 가 저장한 scene_encoder.pt
    vjepa_model_name: str = "facebook/vjepa2-vitg-fpc64-256",
    success_threshold: float = 2.0,
) -> IDEA2Model:
    if camera_names is None:
        camera_names = ["agentview", "eye_in_hand"]
    if obs_tok_len is None:
        obs_tok_len = len(camera_names)

    # ── Obs encoder (ResNet, Mamba 입력용) ────────────────────────────────
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

    # ── SceneEncoder (frozen V-JEPA, z_scene 계산용) ─────────────────────
    scene_encoder = SceneEncoder(
        model_name=vjepa_model_name,
        output_dim=scene_emb_dim,
    ).to(device)
    if scene_encoder_path and os.path.exists(scene_encoder_path):
        scene_encoder.load_state_dict(
            torch.load(scene_encoder_path, map_location=device)
        )
        log.info(f"Loaded SceneEncoder: {scene_encoder_path}")
    else:
        if scene_encoder_path:
            log.warning(f"SceneEncoder path not found: {scene_encoder_path}. Using random proj.")
        else:
            log.warning("No scene_encoder_path provided. Using random proj. "
                        "Run precompute_resnet.py first for consistency.")
    scene_encoder.eval()
    for p in scene_encoder.parameters():
        p.requires_grad = False

    # ── VQ-VAE (Stage 1에서 학습, frozen) ────────────────────────────────
    vqvae = VQVAEModel(
        input_dim=scene_emb_dim,
        hidden_dim=scene_emb_dim,
        K=K,
    ).to(device)
    if vqvae_path and os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        log.info(f"Loaded VQ-VAE: {vqvae_path}")
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # ── Mamba backbone ───────────────────────────────────────────────────
    mamba_encoder = create_mamba_backbone(
        embed_dim=embed_dim,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        device=device,
    )

    # ── LatentVLA policy (Stage 2에서 학습) ──────────────────────────────
    policy = LatentVLAPolicy(
        encoder=mamba_encoder,
        latent_dim=latent_dim,
        lang_emb_dim=lang_emb_dim,
        K=K,
        embed_dim=embed_dim,
        obs_tok_len=obs_tok_len,
    ).to(device)

    # ── Action decoder (Stage 3에서 학습) ────────────────────────────────
    action_decoder = ActionDecoder(
        codebook_dim=scene_emb_dim,
        scene_dim=scene_emb_dim,
        action_seq_len=action_seq_len,
        action_dim=action_dim,
    ).to(device)

    # ── IDEA2Model ───────────────────────────────────────────────────────
    model = IDEA2Model(
        policy=policy,
        obs_encoder=obs_encoder,
        scene_encoder=scene_encoder,
        vqvae=vqvae,
        action_decoder=action_decoder,
        action_seq_len=action_seq_len,
        action_dim=action_dim,
        perception_seq_len=perception_seq_len,
        cam_names=camera_names,
        device=device,
        success_threshold=success_threshold,
    )

    return model


def create_context_encoder(scene_dim: int = 256) -> ContextEncoder:
    """ContextEncoder 생성 (파라미터 없음)."""
    return ContextEncoder(scene_dim=scene_dim)
