"""
GT object localization and phase detection from LIBERO simulation state.

loc_token: 시뮬레이터의 정확한 물체 3D 위치를 카메라 투영으로 변환 → 학습/평가 위치 모두 정확
phase    : gripper state + object height + eef distance → 현재 task phase 자동 감지
"""

import re
import logging
import numpy as np

log = logging.getLogger(__name__)

_FALLBACK_LOC = 512   # 이미지 중심 (물체 못 찾을 때 fallback)

# Phase IDs — action_type 대신 사용
REACH     = 0   # gripper open, eef → object
GRASP     = 1   # gripper closing, eef ≈ object
TRANSPORT = 2   # gripper closed, object lifted
PLACE     = 3   # gripper closed, near target
RELEASE   = 4   # gripper opening, object at target

PHASE_NAMES = {REACH: "reach", GRASP: "grasp", TRANSPORT: "transport",
               PLACE: "place", RELEASE: "release"}
NUM_PHASES  = 5


# ── Object name extraction ──────────────────────────────────────────────────

def extract_object_name(task_name: str) -> str:
    """
    LIBERO task 이름 / instruction에서 target object 이름 추출.

    "pick_up_the_alphabet_soup_and_place_it_in_the_basket" → "alphabet_soup"
    "open_the_top_drawer_of_the_cabinet"                   → "top_drawer"
    """
    s = task_name.lower().replace(" ", "_")

    # pick_up_the_X_and_... or pick_up_the_X (end)
    m = re.search(r'pick_up_the_(.+?)(?:_and_|$)', s)
    if m:
        return m.group(1)

    # open/close/push/stack_the_X
    m = re.search(r'(?:open|close|push|stack|turn)_the_(.+?)(?:_and_|_on_|_to_|_of_|$)', s)
    if m:
        return m.group(1)

    # generic: take first noun phrase after "the"
    parts = s.split('_')
    try:
        idx = parts.index('the')
        stop = {'and', 'in', 'on', 'to', 'from', 'onto', 'of'}
        name_parts = []
        for w in parts[idx + 1:]:
            if w in stop:
                break
            name_parts.append(w)
        if name_parts:
            return '_'.join(name_parts)
    except ValueError:
        pass

    return "object"


# ── Simulation state helpers ────────────────────────────────────────────────

def get_body_pos(sim, name_fragment: str) -> np.ndarray | None:
    """name_fragment를 포함하는 첫 번째 MuJoCo body의 world position 반환."""
    for i in range(sim.model.nbody):
        body_name = sim.model.body_id2name(i)
        if name_fragment.lower() in body_name.lower():
            return sim.data.body_xpos[i].copy()
    return None


def world_to_loc_token(
    sim,
    world_pos: np.ndarray,
    camera: str = "agentview",
    img_h: int = 128,
    img_w: int = 128,
) -> int:
    """
    3D world position → 32×32 grid loc_token (0–1023).
    robosuite camera projection 사용.
    """
    from robosuite.utils.camera_utils import (
        get_camera_transform_matrix,
        project_points_from_world_to_camera,
    )

    T = get_camera_transform_matrix(sim, camera, img_h, img_w)
    pixels = project_points_from_world_to_camera(
        world_pos[np.newaxis], T, img_h, img_w
    )  # → [1, 2]: (pixel_row, pixel_col)

    pixel_row = int(pixels[0, 0])
    pixel_col = int(pixels[0, 1])

    row_32 = max(0, min(31, int(pixel_row / img_h * 31)))
    col_32 = max(0, min(31, int(pixel_col / img_w * 31)))

    return row_32 * 32 + col_32


def get_gt_target_loc_token(
    env,
    task_name: str,
    camera: str = "agentview",
    img_h: int = 128,
    img_w: int = 128,
) -> int:
    """
    시뮬레이터 GT state에서 target placement의 loc_token 계산.
    (basket, drawer 등 물체를 놓을 목표 위치)
    """
    target_name = PhaseDetector._extract_target(task_name)
    world_pos   = get_body_pos(env.sim, target_name)

    if world_pos is None:
        log.debug(f"Target '{target_name}' not found in sim bodies. Using fallback.")
        return _FALLBACK_LOC

    loc_token = world_to_loc_token(env.sim, world_pos, camera, img_h, img_w)
    log.debug(f"GT target: name='{target_name}' pos={world_pos.round(3)} → loc_token={loc_token}")
    return loc_token


def get_gt_loc_token(
    env,
    task_name: str,
    camera: str = "agentview",
    img_h: int = 128,
    img_w: int = 128,
) -> int:
    """
    시뮬레이터 GT state에서 target object의 loc_token 계산.
    물체 위치가 학습과 달라도 항상 정확.
    """
    obj_name = extract_object_name(task_name)
    world_pos = get_body_pos(env.sim, obj_name)

    if world_pos is None:
        log.debug(f"Object '{obj_name}' not found in sim bodies. Using fallback.")
        return _FALLBACK_LOC

    loc_token = world_to_loc_token(env.sim, world_pos, camera, img_h, img_w)
    log.debug(f"GT: obj='{obj_name}' pos={world_pos.round(3)} → loc_token={loc_token}")
    return loc_token


# ── Phase detection ──────────────────────────────────────────────────────────

class PhaseDetector:
    """
    매 스텝 sim state + obs로 현재 task phase 감지.

    reach → grasp → transport → place → release

    training annotation (label_phases_from_demo)과 동일한 로직:
      - gripper open/close: abs(robot0_gripper_qpos).sum() vs threshold
      - lift: ee_pos[z] > grasp_z + LIFT_HEIGHT (grasp_z = eef z at first close)
    robot0_gripper_qpos 스케일(max≈0.042)이 HDF5 gripper_states(max≈0.080)의 약 0.52배
    → open_thr = 0.073 * 0.52 ≈ 0.038
    """

    GRIPPER_OPEN_THR = 0.073  # abs(robot0_gripper_qpos).sum() >= this → open (open≈0.079, grasped≈0.064)
    LIFT_HEIGHT      = 0.02   # grasp_z 대비 eef 상승량 (m)
    PLACE_DIST_XY    = 0.08   # target xy까지 거리 (m)
    EEF_NEAR_OBJ     = 0.06   # eef ↔ object 거리 (m)

    def __init__(self, env, task_name: str, obs_init: dict):
        self.env       = env
        self.task_name = task_name
        self.obj_name  = extract_object_name(task_name)

        obj_pos0 = get_body_pos(env.sim, self.obj_name)
        self.initial_obj_z = obj_pos0[2] if obj_pos0 is not None else 0.85

        self.target_name = self._extract_target(task_name)
        self.phase   = REACH
        self.grasp_z = None   # 처음 gripper 닫힌 시점 eef z

    def step(self, obs: dict) -> int:
        """현재 obs 기반으로 phase를 업데이트하고 반환."""
        openness = float(np.abs(obs["robot0_gripper_qpos"]).sum())
        is_open  = openness >= self.GRIPPER_OPEN_THR

        eef_pos    = obs.get("robot0_eef_pos")
        obj_pos    = get_body_pos(self.env.sim, self.obj_name)
        if obj_pos is None:
            return self.phase

        near_obj = (
            np.linalg.norm(eef_pos - obj_pos) < self.EEF_NEAR_OBJ
            if eef_pos is not None else False
        )
        target_pos  = get_body_pos(self.env.sim, self.target_name)
        near_target = (
            np.linalg.norm(obj_pos[:2] - target_pos[:2]) < self.PLACE_DIST_XY
            if target_pos is not None else False
        )

        if is_open:
            self.grasp_z = None
            if self.phase in (TRANSPORT, PLACE) and near_target:
                self.phase = RELEASE
            elif near_obj:
                self.phase = GRASP
            else:
                self.phase = REACH
        else:
            if self.grasp_z is None and eef_pos is not None:
                self.grasp_z = float(eef_pos[2])

            is_lifted = (
                eef_pos is not None and self.grasp_z is not None
                and float(eef_pos[2]) > self.grasp_z + self.LIFT_HEIGHT
            )
            if is_lifted:
                self.phase = PLACE if near_target else TRANSPORT
            else:
                self.phase = GRASP

        return self.phase

    @staticmethod
    def _extract_target(task_name: str) -> str:
        """task 이름에서 target placement object 추출 (basket, drawer, …)."""
        s = task_name.lower().replace(" ", "_")
        # "place_it_in_the_X" / "into_the_X" / "on_the_X"
        m = re.search(r'(?:in_the|into_the|on_the|onto_the)_(.+?)(?:_and_|$)', s)
        if m:
            return m.group(1)
        return "basket"


# ── Offline phase labeling (for retraining) ─────────────────────────────────

def label_phases_from_demo(
    gripper_states: np.ndarray,  # [T, 2]  절대 finger 위치 (LIBERO HDF5 형식)
    ee_pos:         np.ndarray,  # [T, 3]  end-effector world 위치
    open_thr:    float = 0.073,  # openness(abs sum) >= thr → gripper open
    lift_height: float = 0.02,   # 처음 닫힌 z 기준 상승량 (m)
    window:      int   = 3,      # 평균 z 변화로 하강 감지
) -> np.ndarray:
    """
    LIBERO HDF5 demo trajectory에서 rule-based phase 자동 레이블링.
    물체 위치 없이 gripper_states + ee_pos만 사용 → 수동 레이블링 불필요.

    관측:
      openness (abs sum of finger positions)
        ~0.079 = fully open
        ~0.063 = grasped (object in gripper stops closure)
        threshold 0.073 = open vs closed 구분

    Returns phase IDs [T] with values 0-4.
    """
    T        = len(gripper_states)
    phases   = np.zeros(T, dtype=np.int64)
    openness = np.abs(gripper_states).sum(axis=1)   # [T]
    phase    = REACH
    grasp_z  = None   # gripper 처음 닫힌 시점 z (lift 기준)

    for t in range(T):
        is_open = openness[t] >= open_thr

        if is_open:
            phase   = RELEASE if phase in (TRANSPORT, PLACE) else REACH
            grasp_z = None   # 열리면 기준 리셋
        else:
            if grasp_z is None:
                grasp_z = ee_pos[t, 2]   # 처음 닫힌 z를 기준으로 삼음

            is_lifted = ee_pos[t, 2] > grasp_z + lift_height

            if phase in (TRANSPORT, PLACE):
                # sticky: transport 진입 후 RELEASE 전까지 유지
                if is_lifted:
                    if t >= window:
                        descending = float(ee_pos[t - window:t, 2].mean()) > ee_pos[t, 2] + 0.002
                    else:
                        descending = False
                    phase = PLACE if descending else TRANSPORT
                else:
                    phase = PLACE   # 목적지 높이까지 내려온 상태
            elif is_lifted:
                phase = TRANSPORT
            else:
                phase = GRASP

        phases[t] = phase

    return phases
