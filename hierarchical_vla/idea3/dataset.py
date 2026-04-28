"""
IDEA3 CompDataset.

LIBERO HDF5 데이터에서 (image, lang_emb, comp_delta, actions) 반환.

comp_delta 시뮬레이션:
  - 50% 확률: zero  (첫 번째 primitive 상황 — 환경 변화 없음)
  - 50% 확률: obs_embed[t] - obs_embed[t_ref]
              (t_ref = demo 내 랜덤 이전 시점 → 환경이 바뀐 상황 시뮬)

img_encoder로 obs_embed를 __init__에서 미리 캐싱해 학습 속도 최적화.
"""

import logging
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)

CAM_KEYS = {
    "agentview_image":  "agentview_rgb",
    "eye_in_hand_image": "eye_in_hand_rgb",
}


class CompDataset(Dataset):
    """
    Returns:
        agentview   : (1, C, H, W) float32
        eye_in_hand : (1, C, H, W) float32
        lang_emb    : (512,)
        comp_delta  : (num_cams, latent_dim)  — zero or random delta
        actions     : (action_seq_len, 7)
    """

    def __init__(
        self,
        data_dir: str,
        lang_embs: dict,
        img_encoder: nn.Module,
        device: str = "cpu",
        action_seq_len: int = 10,
        demos_per_task: int = 50,
        comp_prob: float = 0.5,      # comp_delta ≠ 0인 비율
        ref_horizon: int = 50,       # delta 참조 시점 최대 간격
    ):
        self.action_seq_len = action_seq_len
        self.comp_prob = comp_prob
        self.ref_horizon = ref_horizon

        # 샘플 목록: (images_agentview, images_eye, lang_emb, obs_embeds, actions)
        self.samples = []  # per-demo
        self.index = []    # (demo_idx, t) 플랫 인덱스

        file_list = sorted(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))
        log.info(f"Found {len(file_list)} task files in {data_dir}")

        img_encoder = img_encoder.to(device)
        img_encoder.eval()

        for hdf5_file in tqdm(file_list, desc="Loading demos"):
            task_name = hdf5_file.replace("_demo.hdf5", "")
            if task_name not in lang_embs:
                log.warning(f"No lang_emb for {task_name}. Skipping.")
                continue

            lang_emb = lang_embs[task_name]
            if not isinstance(lang_emb, torch.Tensor):
                lang_emb = torch.tensor(lang_emb, dtype=torch.float32)

            path = os.path.join(data_dir, hdf5_file)
            with h5py.File(path, "r") as f:
                demo_keys = list(f["data"].keys())
                indices = np.argsort([int(k[5:]) for k in demo_keys])[:demos_per_task]

                for i in indices:
                    dk = demo_keys[i]
                    grp = f["data"][dk]

                    imgs_agent = grp["obs"]["agentview_rgb"][:]   # (T, H, W, C)
                    imgs_eye   = grp["obs"]["eye_in_hand_rgb"][:] # (T, H, W, C)
                    actions    = grp["actions"][:]                 # (T, 7)
                    T = len(actions)

                    if T < action_seq_len + 1:
                        continue

                    # obs_embed 캐싱 ────────────────────────────────────────
                    obs_embeds = []
                    with torch.no_grad():
                        for t in range(T):
                            ag = _to_tensor(imgs_agent[t], device)   # (1, C, H, W)
                            ey = _to_tensor(imgs_eye[t], device)
                            emb = img_encoder({
                                "agentview_image":   ag,
                                "eye_in_hand_image": ey,
                            })  # (1, num_cams, latent_dim)
                            obs_embeds.append(emb.squeeze(0).cpu())  # (num_cams, D)

                    obs_embeds = torch.stack(obs_embeds, dim=0)  # (T, num_cams, D)

                    demo_idx = len(self.samples)
                    self.samples.append({
                        "imgs_agent": imgs_agent,  # (T, H, W, C) uint8
                        "imgs_eye":   imgs_eye,
                        "lang_emb":   lang_emb,
                        "obs_embeds": obs_embeds,  # (T, num_cams, D) float32
                        "actions":    torch.tensor(actions, dtype=torch.float32),
                    })

                    # 유효 인덱스: t+action_seq_len < T 가 되는 시점만
                    for t in range(T - action_seq_len):
                        self.index.append((demo_idx, t))

        log.info(f"CompDataset: {len(self.index)} samples from {len(self.samples)} demos")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        demo_idx, t = self.index[idx]
        demo = self.samples[demo_idx]

        T = len(demo["actions"])

        # 이미지 (current frame)
        agentview   = _to_tensor(demo["imgs_agent"][t])   # (1, C, H, W)
        eye_in_hand = _to_tensor(demo["imgs_eye"][t])

        # 언어 임베딩
        lang_emb = demo["lang_emb"]  # (512,)

        # action chunk
        actions = demo["actions"][t: t + self.action_seq_len]  # (T_a, 7)

        # comp_delta ────────────────────────────────────────────────────────
        if random.random() < self.comp_prob and t > 0:
            # 랜덤 참조 시점: t_ref ∈ [max(0, t-ref_horizon), t)
            t_ref = random.randint(max(0, t - self.ref_horizon), t - 1)
            comp_delta = demo["obs_embeds"][t] - demo["obs_embeds"][t_ref]  # (num_cams, D)
        else:
            num_cams, D = demo["obs_embeds"].shape[1], demo["obs_embeds"].shape[2]
            comp_delta = torch.zeros(num_cams, D)

        return agentview, eye_in_hand, lang_emb, comp_delta, actions


def _to_tensor(img_np: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """(H, W, C) uint8 → (1, C, H, W) float32 [0,1]"""
    return (
        torch.from_numpy(img_np).float()
        .permute(2, 0, 1).unsqueeze(0) / 255.0
    ).to(device)
