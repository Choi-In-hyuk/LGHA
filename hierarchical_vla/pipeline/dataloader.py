"""
HierarchicalLiberoDataset: 기존 LiberoDataset에 annotation 파일을 얹어
(obs, action_type, loc_token, action) 형식으로 반환한다.

annotation 파일이 없으면 action_type=0, loc_token=512(중심)으로 fallback.

phase 모드 (use_phase=True):
  annotation에 phase_labels가 있으면 per-frame phase를 action_type으로 사용.
  5단계: 0=reach, 1=grasp, 2=transport, 3=place, 4=release
  이 경우 num_action_types=5 로 모델 초기화 필요.
"""

import logging
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

_CENTER_LOC = 512


class HierarchicalLiberoDataset(Dataset):
    """
    반환 형식:
        obs = {
            "agentview_image":   [1, C, H, W]  float32 [0,1]
            "eye_in_hand_image": [1, C, H, W]  float32 [0,1]
            "robot_states":      [1, state_dim] float32
        }
        action_type : scalar int64  (phase 모드: per-frame 0-4, 레거시: task 기반 0-15)
        loc_token   : scalar int64
        actions     : [chunk_size, action_dim] float32
        mask        : [chunk_size] float32
    """

    def __init__(
        self,
        data_directory: str,
        annotation_path: str = None,
        max_len_data: int = 530,
        chunk_size: int = 10,
        start_idx: int = 0,
        demos_per_task: int = 50,
        action_dim: int = 7,
        state_dim: int = 9,          # joint(7) + gripper(2)
        use_phase: bool = True,       # True → phase_labels를 action_type으로 사용
    ):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.state_dim  = state_dim
        self.use_phase  = use_phase
        self.camera_names = ["agentview", "eye_in_hand"]

        # annotation 로드
        annotations = {}
        if annotation_path and os.path.exists(annotation_path):
            with open(annotation_path, "rb") as f:
                annotations = pickle.load(f)
            # phase_labels 유무 확인
            sample = next(iter(annotations.values()), {})
            has_phase = "phase_labels" in sample
            if use_phase and not has_phase:
                log.warning("use_phase=True 이지만 annotation에 phase_labels 없음. action_type 사용.")
                self.use_phase = False
            log.info(
                f"Loaded {len(annotations)} annotations "
                f"(phase={'yes' if has_phase else 'no'})"
            )
        else:
            log.warning(
                f"Annotation file not found: {annotation_path}. "
                "Using fallback (action_type=0, loc_token=center)."
            )
            self.use_phase = False

        # HDF5 파일 순회
        self._slices = []
        self._demos  = []

        hdf5_files = sorted(f for f in os.listdir(data_directory) if f.endswith(".hdf5"))
        for hdf5_file in hdf5_files:
            task_name = os.path.splitext(hdf5_file)[0]
            fpath = os.path.join(data_directory, hdf5_file)

            with h5py.File(fpath, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=lambda k: int(k[5:]))
                demo_keys = demo_keys[start_idx: start_idx + demos_per_task]

                for demo_key in demo_keys:
                    demo = f["data"][demo_key]
                    T    = int(demo.attrs.get("num_samples", len(demo["actions"])))

                    agentview_rgb   = demo["obs"]["agentview_rgb"][:]    # [T,H,W,C]
                    eye_in_hand_rgb = demo["obs"]["eye_in_hand_rgb"][:]  # [T,H,W,C]

                    joint_states   = demo["obs"]["joint_states"][:]      # [T,7]
                    gripper_states = demo["obs"]["gripper_states"][:]    # [T,2]
                    robot_states   = np.concatenate([joint_states, gripper_states], axis=-1)

                    # actions: pad to max_len_data
                    action_data  = demo["actions"][:]
                    zero_actions = np.zeros((max_len_data, action_dim), dtype=np.float32)
                    zero_mask    = np.zeros((max_len_data,),             dtype=np.float32)
                    zero_actions[:T] = action_data
                    zero_mask[:T]    = 1.0

                    # annotation lookup
                    ann_key = f"{task_name}_{demo_key}"
                    ann     = annotations.get(ann_key, {})

                    action_type  = int(ann.get("action_type", 0))
                    loc_tokens   = ann.get("loc_tokens", [_CENTER_LOC] * T)
                    phase_labels = ann.get("phase_labels", None)

                    if len(loc_tokens) < T:
                        loc_tokens = loc_tokens + [loc_tokens[-1]] * (T - len(loc_tokens))

                    # phase_labels: per-frame, pad if needed
                    if phase_labels is not None:
                        if len(phase_labels) < T:
                            phase_labels = phase_labels + [phase_labels[-1]] * (T - len(phase_labels))
                    else:
                        phase_labels = [action_type] * T   # fallback: 모든 스텝 동일 action_type

                    demo_idx = len(self._demos)
                    self._demos.append({
                        "agentview_rgb":   agentview_rgb,
                        "eye_in_hand_rgb": eye_in_hand_rgb,
                        "robot_states":    robot_states,
                        "actions":         torch.from_numpy(zero_actions),
                        "mask":            torch.from_numpy(zero_mask),
                        "action_type":     action_type,
                        "phase_labels":    phase_labels,   # [T] per-frame
                        "loc_tokens":      loc_tokens,
                        "T":               T,
                    })

                    for s in range(T - chunk_size + 1):
                        self._slices.append((demo_idx, s, s + chunk_size))

        log.info(f"Dataset: {len(self._demos)} demos, {len(self._slices)} slices")

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, idx):
        demo_idx, start, end = self._slices[idx]
        d = self._demos[demo_idx]

        # 현재 스텝 프레임
        t = start
        agentview  = torch.from_numpy(d["agentview_rgb"][t]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        eye        = torch.from_numpy(d["eye_in_hand_rgb"][t]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        robot_st   = torch.from_numpy(d["robot_states"][t]).float().unsqueeze(0)

        obs = {
            "agentview_image":   agentview,
            "eye_in_hand_image": eye,
            "robot_states":      robot_st,
        }

        # phase 모드: per-frame phase 사용, 레거시: task-level action_type
        at_val      = d["phase_labels"][t] if self.use_phase else d["action_type"]
        action_type = torch.tensor(at_val,             dtype=torch.long)
        loc_token   = torch.tensor(d["loc_tokens"][t], dtype=torch.long)
        actions     = d["actions"][start:end]
        mask        = d["mask"][start:end]

        return obs, action_type, loc_token, actions, mask

    def get_all_actions(self):
        result = []
        for d in self._demos:
            T = d["T"]
            result.append(d["actions"][:T])
        return torch.cat(result, dim=0)
