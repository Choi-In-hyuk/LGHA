"""
IDEA3 EpisodeMemory.

obs_embed delta(primitive 전후 차이)를 저장하고
누적 compensation을 제공한다.
"""

import json
import os
import torch


class EpisodeMemory:
    def __init__(self, path: str = None):
        self.path = path
        self.history = []   # list of {"primitive", "success"}
        self.deltas = []    # list of (num_cams, latent_dim) tensors

    def reset(self):
        self.history = []
        self.deltas = []

    def write(self, primitive: str, delta_obs: torch.Tensor, success: bool):
        """
        primitive 완료 후 호출.
        delta_obs: (num_cams, latent_dim) — obs_embed_after - obs_embed_before
        """
        self.deltas.append(delta_obs.detach().cpu())
        self.history.append({"primitive": primitive, "success": success})
        if self.path:
            self._save()

    def get_cumulative_delta(self) -> torch.Tensor | None:
        """지금까지 누적된 delta. 없으면 None."""
        if not self.deltas:
            return None
        return torch.stack(self.deltas).sum(dim=0)  # (num_cams, latent_dim)

    def get_feedback_for_llm(self) -> str:
        lines = []
        for h in self.history:
            status = "succeeded" if h["success"] else "failed"
            lines.append(f"- {h['primitive']}: {status}")
        return "\n".join(lines)

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "history": self.history,
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
