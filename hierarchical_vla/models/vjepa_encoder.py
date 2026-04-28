"""
V-JEPA 2 Scene Encoder

현재 observation을 V-JEPA 2 encoder로 encoding해 semantic scene latent z_scene을 생성.
agentview_image (third-person view)를 입력으로 사용.

Fine-tuning 전략:
  - 하위 레이어(early layers): frozen (general visual features)
  - 상위 레이어(last n_finetune_layers): fine-tuned (task-specific scene understanding)
"""

import torch
import torch.nn as nn
from transformers import VJEPA2Model, VJEPA2Config

import logging
log = logging.getLogger(__name__)


class VJEPASceneEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitg-fpc64-256",
        output_dim: int = 256,
        n_finetune_layers: int = 4,      # 상위 N개 레이어만 fine-tune
        input_size: int = 256,           # V-JEPA 입력 해상도
        n_frames: int = 4,               # 반복할 프레임 수 (단일 이미지 → 짧은 클립)
    ):
        super().__init__()

        self.input_size = input_size
        self.n_frames = n_frames

        log.info(f"Loading V-JEPA 2 encoder: {model_name}")
        self.encoder = VJEPA2Model.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 1408 for ViT-g

        # 전체 frozen 후 상위 N 레이어만 해동
        self._freeze_except_last_n(n_finetune_layers)

        # z_scene → scene_token 프로젝션
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, output_dim),
        )

        # 입력 이미지 리사이즈 (LIBERO 128x128 → 256x256)
        self.resize = nn.Upsample(size=(input_size, input_size), mode="bilinear", align_corners=False)

    def _freeze_except_last_n(self, n: int):
        """전체 파라미터 frozen 후 encoder 마지막 n 레이어만 해동."""
        for param in self.encoder.parameters():
            param.requires_grad = False

        # V-JEPA2 encoder는 encoder.encoder.layer 구조
        layers = self.encoder.encoder.layer
        total = len(layers)
        for layer in layers[total - n:]:
            for param in layer.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder.parameters())
        log.info(f"V-JEPA fine-tune: {trainable:,} / {total_params:,} params ({100*trainable/total_params:.1f}%)")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W) - agentview_image, float32, [0,1]

        Returns:
            z_scene: (B, output_dim)
        """
        B = image.shape[0]

        # 128x128 → 256x256
        image = self.resize(image)  # (B, C, 256, 256)

        # 단일 프레임 → 짧은 클립: (B, C, H, W) → (B, n_frames, C, H, W)
        clip = image.unsqueeze(1).expand(-1, self.n_frames, -1, -1, -1)

        # V-JEPA2 입력 형식: (B, T, C, H, W)
        outputs = self.encoder(
            pixel_values_videos=clip,
            skip_predictor=True,      # encoder 출력만 사용
        )

        # encoder hidden states: (B, num_patches, hidden_size)
        hidden = outputs.last_hidden_state  # (B, N, 1408)

        # mean pooling → (B, 1408)
        z = hidden.mean(dim=1)

        # projection → (B, output_dim)
        z_scene = self.proj(z)  # (B, 256)

        return z_scene
