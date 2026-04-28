"""
SceneEncoder: V-JEPA → z_scene.

V-JEPA의 semantic latent space를 활용해 선형 보정이 잘 동작하도록 함.
z_now - cumulative_delta ≈ z_train (학습 분포 복원)

precompute_scene.py에서 가중치(scene_encoder.pt) 저장 →
이후 모든 단계(Stage 1/2/3/inference)에서 동일 파일 로드 → z_scene 공간 일관성.
"""

import logging
import torch
import torch.nn as nn
from transformers import VJEPA2Model

log = logging.getLogger(__name__)


class SceneEncoder(nn.Module):
    """
    Frozen V-JEPA + frozen linear projection → z_scene (256-dim).

    전체 파라미터 frozen: 학습/inference 완전 동일한 출력.
    proj 가중치는 precompute_scene.py 최초 실행 시 저장,
    이후 모든 단계에서 동일 파일 로드.

    입력: (B, C, H, W) float32 [0,1]
    출력: (B, output_dim)
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitg-fpc64-256",
        output_dim: int = 256,
        input_size: int = 256,
        n_frames: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.n_frames   = n_frames

        log.info(f"Loading V-JEPA: {model_name}")
        self.encoder = VJEPA2Model.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size  # 1408 for ViT-g

        # 전체 frozen
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_dim),
        )

        self.resize = nn.Upsample(
            size=(input_size, input_size), mode="bilinear", align_corners=False
        )

        # proj도 frozen (precompute 시 저장한 가중치와 동일하게 유지)
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W) float32 [0,1]
        Returns:
            z_scene: (B, output_dim)
        """
        image = self.resize(image)
        clip  = image.unsqueeze(1).expand(-1, self.n_frames, -1, -1, -1)

        outputs = self.encoder(
            pixel_values_videos=clip,
            skip_predictor=True,
        )
        hidden  = outputs.last_hidden_state   # (B, N, 1408)
        z       = hidden.mean(dim=1)          # (B, 1408)
        return self.proj(z)                   # (B, output_dim)
