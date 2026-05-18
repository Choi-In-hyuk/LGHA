"""
Qwen2-VL 기반 High-Level Orchestrator.

자연어 명령 + 현재 이미지를 받아 한 번의 LLM 호출로:
  - action_type (int): 수행할 행동 유형
  - loc_token (int):   target object의 center 위치 (0-1023, 32×32 grid)
를 동시에 반환한다.

Qwen2-VL bounding box 출력 형식:
  <|object_ref_start|>object<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
  좌표 범위: [0, 1000] (정규화된 픽셀 × 1000)
"""

import re
import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .action_type_mapper import ACTION_NAME_TO_ID, ACTION_TYPES, NUM_ACTION_TYPES

log = logging.getLogger(__name__)

# Qwen2-VL bounding box 파싱 패턴
_BOX_PATTERN = re.compile(r"\((\d+),(\d+)\),\((\d+),(\d+)\)")

_SYSTEM_PROMPT = f"""You are a robot task planner. Given an image and a task instruction, output exactly two lines:
action: <one of: {", ".join(ACTION_TYPES.values())}>
object: <target object name>

Then on the next line, locate the target object using grounding."""

_USER_TEMPLATE = "Task instruction: {instruction}\n\nIdentify the action type and locate the target object in the image."

_DUAL_LOC_TEMPLATE = (
    "Task: {instruction}\n"
    "Look at the image and output the grid positions (0-1023) of:\n"
    "1) the object to pick up\n"
    "2) the target location to place it\n"
    "Format: object: NNN target: MMM"
)

_CENTER_LOC = 512   # fallback: 이미지 중심


class QwenOrchestrator:
    """
    Qwen2-VL을 이용한 고수준 계획기.
    annotation(오프라인) 및 inference(실시간) 양쪽에서 사용한다.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        batch_size: int = 4,
        lora_path: str = None,
    ):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        log.info(f"Loading Qwen2-VL: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        if lora_path is not None:
            from peft import PeftModel
            log.info(f"Loading LoRA weights: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            log.info("LoRA merged.")

        self.model = model.eval()
        self.device = device
        self.batch_size = batch_size
        self.has_lora = lora_path is not None
        log.info("Qwen2-VL loaded.")

    @torch.no_grad()
    def infer(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> tuple[int, int]:
        """
        단일 프레임 추론. inference 루프에서 사용.

        image      : [H, W, C] uint8 RGB
        instruction: 자연어 명령 (예: "오른쪽에 빨간 거 집어줘")
        Returns    : (action_type_id, loc_token)
        """
        action_type, obj_name = self._get_action_and_object(image, instruction)
        loc_token = self._get_loc_token(image, obj_name)
        return action_type, loc_token

    @torch.no_grad()
    def infer_dual_loc(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> tuple[int, int, int]:
        """
        단일 프레임 추론 (파인튜닝 모델용).
        물체 위치 + 목표 위치를 동시에 반환.

        Returns: (action_type_id, obj_loc_token, target_loc_token)
        """
        action_type, _ = self._get_action_and_object(image, instruction)
        obj_loc, target_loc = self._get_dual_loc(image, instruction)
        return action_type, obj_loc, target_loc

    def _get_dual_loc(self, frame: np.ndarray, instruction: str) -> tuple[int, int]:
        """파인튜닝 모델에서 obj_loc + target_loc 동시 추출."""
        pil = Image.fromarray(frame).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text",  "text": _DUAL_LOC_TEMPLATE.format(
                        instruction=instruction
                    )},
                ],
            }
        ]
        text = self._generate(messages, max_new_tokens=20)
        return _parse_dual_loc(text)

    @torch.no_grad()
    def annotate_frames(
        self,
        frames: np.ndarray,
        instruction: str,
        stride: int = 5,
    ) -> tuple[int, list[int]]:
        """
        오프라인 annotation. 전체 trajectory의 loc_tokens를 추출한다.

        frames     : [T, H, W, C] uint8 RGB
        instruction: task 이름 또는 자연어 명령
        stride     : 매 stride 프레임마다 Qwen2-VL 실행, 나머지는 forward-fill
        Returns    : (action_type_id, loc_tokens[T])
        """
        T = len(frames)

        # action_type은 첫 프레임 한 번만
        action_type, obj_name = self._get_action_and_object(frames[0], instruction)
        log.debug(f"  action={ACTION_TYPES[action_type]}, object='{obj_name}'")

        # loc_token: stride 간격으로 추출 후 보간
        key_indices = list(range(0, T, stride))
        key_locs = []
        for i in range(0, len(key_indices), self.batch_size):
            batch_idx = key_indices[i: i + self.batch_size]
            batch_frames = [frames[j] for j in batch_idx]
            locs = self._batch_get_loc(batch_frames, obj_name)
            key_locs.extend(locs)

        loc_tokens = _forward_fill(key_indices, key_locs, T)
        return action_type, loc_tokens

    # ── 내부 메서드 ──────────────────────────────────────────────────────────

    def _get_action_and_object(
        self, frame: np.ndarray, instruction: str
    ) -> tuple[int, str]:
        """이미지 + 명령 → (action_type_id, object_name)"""
        pil = Image.fromarray(frame).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text",  "text": _USER_TEMPLATE.format(instruction=instruction)},
                ],
            }
        ]
        text = self._generate(messages, max_new_tokens=60)
        return _parse_action_and_object(text)

    def _get_loc_token(self, frame: np.ndarray, obj_name: str) -> int:
        """이미지 + object 이름 → loc_token"""
        pil = Image.fromarray(frame).convert("RGB")
        prompt = (
            f"Output the bounding box of the {obj_name} as four integers: "
            f"x1,y1,x2,y2 (range 0-1000, origin at top-left). "
            f"Output only the four numbers separated by commas, nothing else."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]
        text = self._generate(messages, max_new_tokens=30)
        return _parse_loc_token(text)

    def _batch_get_loc(
        self, frames: list[np.ndarray], obj_name: str
    ) -> list[int]:
        results = []
        for frame in frames:
            results.append(self._get_loc_token(frame, obj_name))
        return results

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        from qwen_vl_utils import process_vision_info

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        trimmed = output_ids[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=False
        )[0]


# ── 파싱 헬퍼 ────────────────────────────────────────────────────────────────

def _parse_action_and_object(text: str) -> tuple[int, str]:
    """
    Qwen2-VL 응답에서 action_type ID와 object 이름을 추출.
    예상 형식:
        action: pick_and_place
        object: butter
    """
    action_id = 0   # fallback: pick_and_place
    obj_name  = "object"

    for line in text.lower().splitlines():
        line = line.strip()
        if line.startswith("action:"):
            val = line.split("action:", 1)[-1].strip()
            # 가장 근접한 action 이름 매핑
            for name, aid in ACTION_NAME_TO_ID.items():
                if name.replace("_", " ") in val or name in val:
                    action_id = aid
                    break
            else:
                # keyword fallback
                from .action_type_mapper import task_name_to_action_type
                action_id = task_name_to_action_type(val)

        elif line.startswith("object:"):
            raw = line.split("object:", 1)[-1].strip()
            # special token 이전까지만 사용
            raw = re.split(r"<\|", raw)[0].strip()
            if raw:
                obj_name = raw

    return action_id, obj_name


def _parse_loc_token(text: str) -> int:
    """
    Qwen2-VL bounding box 출력에서 center loc_token 계산.
    형식 1: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    형식 2: x1,y1,x2,y2  (쉼표 구분 정수)
    좌표 범위: [0, 1000]
    """
    m = _BOX_PATTERN.search(text)
    if m:
        x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    else:
        nums = re.findall(r'\d+', text)
        if len(nums) >= 4:
            x1, y1, x2, y2 = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
        else:
            log.debug(f"bbox parse failed: '{text[:80]}'. Using center.")
            return _CENTER_LOC

    if not (0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000):
        log.debug(f"bbox out of range: {x1},{y1},{x2},{y2}. Using center.")
        return _CENTER_LOC

    # center 좌표 → [0, 1] 정규화
    cx = (x1 + x2) / 2.0 / 1000.0
    cy = (y1 + y2) / 2.0 / 1000.0

    # [0, 1] → 32×32 grid
    col = int(cx * 31)
    row = int(cy * 31)
    return row * 32 + col


def _parse_dual_loc(text: str) -> tuple[int, int]:
    """
    "object: 342 target: 615" 형식 파싱.
    Returns (obj_loc, target_loc).
    """
    obj_loc    = _CENTER_LOC
    target_loc = _CENTER_LOC

    m_obj = re.search(r'object\s*:\s*(\d+)', text, re.IGNORECASE)
    m_tgt = re.search(r'target\s*:\s*(\d+)', text, re.IGNORECASE)

    if m_obj:
        v = int(m_obj.group(1))
        obj_loc = max(0, min(1023, v))
    if m_tgt:
        v = int(m_tgt.group(1))
        target_loc = max(0, min(1023, v))

    if not m_obj or not m_tgt:
        log.debug(f"dual_loc parse fallback: '{text[:80]}'")

    return obj_loc, target_loc


def _forward_fill(key_indices: list[int], key_locs: list[int], T: int) -> list[int]:
    result = [_CENTER_LOC] * T
    for idx, loc in zip(key_indices, key_locs):
        result[idx] = loc
    last = key_locs[0] if key_locs else _CENTER_LOC
    for t in range(T):
        if result[t] != _CENTER_LOC or t == 0:
            last = result[t]
        else:
            result[t] = last
    return result
