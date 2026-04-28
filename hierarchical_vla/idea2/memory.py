"""
EpisodeMemory: primitive 실행 결과를 JSON 파일로 기록.

각 task 시작 시 reset(), primitive 완료마다 write().
LLM 재계획(replan) 시 get_feedback_for_llm()으로 요약 제공.
"""

import json
import logging
import os

import torch

log = logging.getLogger(__name__)


class EpisodeMemory:
    """
    파일 기반 episode 기억 저장소.

    history 항목:
        primitive   (str)         어떤 명령이었는지
        delta_z     (list[float]) 씬이 어떻게 바뀌었는지 (z_after - z_before)
        success     (bool)        성공 여부
        codebook_k  (int)         선택된 행동 패턴 index
    """

    def __init__(self, path: str = None):
        """
        Args:
            path: JSON 저장 경로. None이면 메모리만 사용 (파일 저장 없음).
        """
        self.path = path
        self.history: list[dict] = []

    def reset(self):
        """Task 시작 시 호출. 기록 초기화."""
        self.history = []
        if self.path:
            self._save()
        log.debug("EpisodeMemory reset.")

    def write(
        self,
        primitive: str,
        delta_z: torch.Tensor,
        success: bool,
        codebook_k: int,
    ):
        """
        Primitive 완료 후 호출.

        Args:
            primitive:  primitive ID (str)
            delta_z:    (scene_dim,) z_after - z_before
            success:    성공 여부
            codebook_k: 선택된 codebook index
        """
        entry = {
            "primitive":  primitive,
            "delta_z":    delta_z.float().cpu().tolist(),
            "success":    bool(success),
            "codebook_k": int(codebook_k),
        }
        self.history.append(entry)
        if self.path:
            self._save()
        log.debug(f"Memory: [{primitive}] {'OK' if success else 'FAIL'} k={codebook_k}")

    def get_delta_history(self) -> list:
        """
        지금까지 기록된 delta_z 리스트 반환 (oldest first).
        ContextEncoder.forward()에 바로 넘길 수 있는 형태.
        """
        return [torch.tensor(h["delta_z"]) for h in self.history]

    def get_feedback_for_llm(self) -> str:
        """
        실패 발생 시 LLM에게 넘길 텍스트 요약.
        """
        if not self.history:
            return "No steps executed yet."

        lines = []
        for i, h in enumerate(self.history):
            status = "SUCCESS" if h["success"] else "FAILED"
            delta_norm = torch.tensor(h["delta_z"]).norm().item()
            lines.append(
                f"  Step {i+1} [{h['primitive']}]: {status} "
                f"(scene_change={delta_norm:.3f}, codebook_k={h['codebook_k']})"
            )
        return "\n".join(lines)

    def last_success(self) -> bool:
        """마지막 primitive 성공 여부."""
        if not self.history:
            return True
        return self.history[-1]["success"]

    def _save(self):
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump({"history": self.history}, f, indent=2)

    def load(self):
        """기존 파일에서 history 로드."""
        if self.path and os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.history = data.get("history", [])
