"""
PaliGemma로 각 프레임에서 target object의 loc_token을 추출한다.

PaliGemma detection 출력 형식:
  "<loc_NNN><loc_NNN><loc_NNN><loc_NNN> label"
  순서: y1, x1, y2, x2  (32×32 grid, 0-1023)

center loc_token 계산:
  center_row = (y1_row + y2_row) // 2
  center_col = (x1_col + x2_col) // 2
  center_loc  = center_row * 32 + center_col
"""

import re
import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

LOC_PATTERN = re.compile(r"<loc_(\d+)>")
DEFAULT_LOC = 512   # 이미지 중심 (row=16, col=0 → 실제론 16*32=512)


class LocTokenAnnotator:
    """PaliGemma를 이용해 이미지에서 target object의 center loc_token을 반환한다."""

    def __init__(
        self,
        model_id: str = "google/paligemma-3b-ft-coco-det-224",
        device: str = "cuda",
        batch_size: int = 8,
    ):
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        log.info(f"Loading PaliGemma: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.device = device
        self.batch_size = batch_size
        log.info("PaliGemma loaded.")

    @torch.no_grad()
    def annotate_frames(
        self,
        frames: np.ndarray,
        object_name: str,
        stride: int = 1,
    ) -> list[int]:
        """
        frames: [T, H, W, C] uint8 RGB
        object_name: PaliGemma에 넘길 object 이름 (예: "butter", "drawer")
        stride: 매 stride 프레임마다 PaliGemma 실행, 나머지는 보간

        Returns: loc_tokens [T], 각 프레임의 center loc_token (0-1023)
        """
        T = len(frames)
        prompt = f"detect {object_name}"

        # stride로 처리할 인덱스 선택
        key_indices = list(range(0, T, stride))
        key_frames = [frames[i] for i in key_indices]

        key_locs = self._run_paligemma(key_frames, prompt)

        # 전체 프레임에 선형 보간 (stride=1이면 그대로)
        loc_tokens = _interpolate_locs(key_indices, key_locs, T)
        return loc_tokens

    def _run_paligemma(self, frames: list[np.ndarray], prompt: str) -> list[int]:
        results = []
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i: i + self.batch_size]
            pil_images = [Image.fromarray(f).convert("RGB") for f in batch_frames]

            inputs = self.processor(
                text=[prompt] * len(pil_images),
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

            decoded = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=False,
            )

            for text in decoded:
                results.append(_parse_center_loc(text))

        return results


def _parse_center_loc(text: str) -> int:
    """
    PaliGemma 출력 텍스트에서 bounding box center의 loc_token을 계산한다.
    출력 예: "<loc_0342><loc_0128><loc_0512><loc_0256> butter"
             순서: y1, x1, y2, x2
    """
    locs = LOC_PATTERN.findall(text)
    if len(locs) < 4:
        # detection 실패 — 이미지 중심 반환
        log.debug(f"PaliGemma parse failed: '{text}'. Using center.")
        return DEFAULT_LOC

    y1, x1, y2, x2 = int(locs[0]), int(locs[1]), int(locs[2]), int(locs[3])

    # loc_value → 32×32 grid 좌표
    y1_row, x1_col = y1 // 32, y1 % 32
    y2_row, x2_col = y2 // 32, y2 % 32

    center_row = (y1_row + y2_row) // 2
    center_col = (x1_col + x2_col) // 2
    return int(center_row * 32 + center_col)


def _interpolate_locs(key_indices: list[int], key_locs: list[int], T: int) -> list[int]:
    """key_indices의 loc_token을 T 프레임 전체에 보간한다."""
    if len(key_indices) == T:
        return key_locs

    result = [DEFAULT_LOC] * T
    for i, (idx, loc) in enumerate(zip(key_indices, key_locs)):
        result[idx] = loc

    # 앞/뒤 채우기
    last = key_locs[0]
    for t in range(T):
        if t in dict(zip(key_indices, key_locs)):
            last = result[t]
        else:
            result[t] = last

    return result
