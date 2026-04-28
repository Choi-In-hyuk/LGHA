"""
LLM Orchestrator for LIBERO-10 evaluation.

GPT-4o decomposes a LIBERO-10 multi-step task into a sequence of
LIBERO-90 primitives. MambaVLA executes each primitive with a
home-pose return between calls.
"""

import json
import logging
import os
import numpy as np
import torch
from openai import OpenAI

log = logging.getLogger(__name__)

# Home pose controller gain and steps
HOME_GAIN = 3.0   # clip되어 사실상 max speed
HOME_STEPS = 60   # position correction steps


def _get_scene_key(task_name: str) -> str:
    """Extract scene key from task name.
    e.g. KITCHEN_SCENE3_turn_on_the_stove -> KITCHEN_SCENE3
         LIVING_ROOM_SCENE2_pick_up_...   -> LIVING_ROOM_SCENE2
    """
    parts = task_name.split("_")
    if parts[0] == "LIVING":
        # LIVING_ROOM_SCENE2_... -> LIVING_ROOM_SCENE2
        return "LIVING_ROOM_" + parts[2]
    return parts[0] + "_" + parts[1]


def _load_primitives(primitives_path: str) -> dict:
    with open(primitives_path, "r") as f:
        return json.load(f)


def build_system_prompt(scene_primitives: list[dict]) -> str:
    primitive_list = "\n".join(
        f"  - id: \"{p['id']}\"\n    description: \"{p['description']}\""
        for p in scene_primitives
    )
    return f"""You are a robot brain that plans manipulation tasks.
You must decompose a given task into an ordered list of primitive actions.
Only use primitives from the list below. Do NOT invent new primitives.

Available primitives for this scene:
{primitive_list}

Respond with a JSON array of primitive IDs in execution order.
Example: ["KITCHEN_SCENE3_turn_on_the_stove", "KITCHEN_SCENE3_put_the_frying_pan_on_it"]
Respond with the JSON array only, no explanation."""


def build_replan_system_prompt(scene_primitives: list[dict]) -> str:
    primitive_list = "\n".join(
        f"  - id: \"{p['id']}\"\n    description: \"{p['description']}\""
        for p in scene_primitives
    )
    return f"""You are a robot brain that replans manipulation tasks after partial failures.
You will be given the execution history and must decide on the remaining steps.
Only use primitives from the list below. Do NOT invent new primitives.

Available primitives for this scene:
{primitive_list}

Respond with a JSON array of primitive IDs for the REMAINING steps only.
Respond with the JSON array only, no explanation."""


def decompose_task(
    client: OpenAI,
    task_name: str,
    scene_primitives: list[dict],
    model: str = "gpt-4o",
) -> list[str]:
    """
    Call GPT-4o to decompose a LIBERO-10 task into LIBERO-90 primitive IDs.

    Returns a list of primitive task IDs in execution order.
    """
    task_description = " ".join(task_name.split("_")[2:])  # strip scene prefix
    system_prompt = build_system_prompt(scene_primitives)
    user_message = f'Decompose this task into primitives: "{task_description}"'

    log.info(f"LLM decomposing: {task_description}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    log.info(f"LLM response: {raw}")

    # Parse JSON array
    primitive_ids = json.loads(raw)

    # Validate each ID exists in primitives
    valid_ids = {p["id"] for p in scene_primitives}
    for pid in primitive_ids:
        if pid not in valid_ids:
            log.warning(f"LLM returned unknown primitive: {pid}. Skipping.")
    primitive_ids = [pid for pid in primitive_ids if pid in valid_ids]

    log.info(f"Decomposed into {len(primitive_ids)} primitives: {primitive_ids}")
    return primitive_ids


def go_to_home_pose(env, obs, home_eef_pos, home_eef_quat, home_gripper,
                    record: bool = False):
    """
    3단계 home pose return:
      1. 그리퍼 열기 (10 steps)
      2. 위치 보정 (HOME_STEPS)
      3. 회전 보정 (30 steps)
    """
    frames = []

    # Step 1: 그리퍼 열기 — LIBERO: -1.0 = open
    for _ in range(10):
        action = np.concatenate([np.zeros(3), np.zeros(3), [-1.0]])
        obs, _, _, _ = env.step(action)
        if record:
            frames.append(obs["agentview_image"].copy())

    # Step 2: 위치 보정
    for _ in range(HOME_STEPS):
        current_pos = obs["robot0_eef_pos"]
        delta_pos = np.clip((home_eef_pos - current_pos) * HOME_GAIN, -1.0, 1.0)
        action = np.concatenate([delta_pos, np.zeros(3), [-1.0]])
        obs, _, _, _ = env.step(action)
        if record:
            frames.append(obs["agentview_image"].copy())

    # Step 3: 회전 보정
    for _ in range(30):
        delta_rot = np.clip(
            _quat_to_delta_rot(home_eef_quat, obs["robot0_eef_quat"]) * 0.2,
            -0.3, 0.3
        )
        action = np.concatenate([np.zeros(3), delta_rot, [-1.0]])
        obs, _, _, _ = env.step(action)
        if record:
            frames.append(obs["agentview_image"].copy())

    return obs, frames


def _quat_to_delta_rot(target_quat, current_quat):
    """
    Compute axis-angle rotation error: target * inv(current).
    Quaternion format: [x, y, z, w] (robosuite convention).
    """
    cw, cx, cy, cz = current_quat[3], current_quat[0], current_quat[1], current_quat[2]
    tw, tx, ty, tz = target_quat[3], target_quat[0], target_quat[1], target_quat[2]

    # target * inv(current),  inv(q) = [w, -x, -y, -z]
    ew = tw * cw + tx * cx + ty * cy + tz * cz
    ex = tx * cw - tw * cx - ty * cz + tz * cy
    ey = ty * cw - tw * cy + tx * cz - tz * cx
    ez = tz * cw - tw * cz - tx * cy + ty * cx

    sin_half = np.sqrt(ex**2 + ey**2 + ez**2)
    if sin_half < 1e-6:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, abs(ew))
    axis = np.array([ex, ey, ez]) / sin_half
    return axis * angle


class LLMOrchestrator:
    """
    High-level orchestrator: GPT-4o plans → MambaVLA executes each primitive
    with home-pose return in between.
    """

    def __init__(
        self,
        primitives_path: str,
        openai_api_key: str,
        steps_per_primitive: int = 150,
        gpt_model: str = "gpt-4o",
        device: str = "cuda",
    ):
        self.primitives = _load_primitives(primitives_path)
        self.client = OpenAI(api_key=openai_api_key)
        self.steps_per_primitive = steps_per_primitive
        self.gpt_model = gpt_model
        self.device = device

        # Cache decompositions to avoid redundant API calls
        self._decomp_cache: dict[str, list[str]] = {}

    def get_primitives_for_task(self, task_name: str) -> list[dict]:
        scene_key = _get_scene_key(task_name)
        return self.primitives.get(scene_key, [])

    def decompose(self, task_name: str) -> list[str]:
        if task_name in self._decomp_cache:
            return self._decomp_cache[task_name]

        scene_primitives = self.get_primitives_for_task(task_name)
        if not scene_primitives:
            log.warning(f"No primitives found for scene of task: {task_name}")
            return [task_name]

        primitive_ids = decompose_task(
            self.client, task_name, scene_primitives, model=self.gpt_model
        )

        if not primitive_ids:
            log.warning("LLM returned empty plan. Falling back to full task name.")
            primitive_ids = [task_name]

        self._decomp_cache[task_name] = primitive_ids
        return primitive_ids

    def replan(
        self,
        task_name: str,
        memory_feedback: str,
        remaining_primitives: list[str],
    ) -> list[str]:
        """
        실패 발생 시 LLM에 현재 상태를 알리고 나머지 계획을 재수립.

        Args:
            task_name:           현재 task
            memory_feedback:     EpisodeMemory.get_feedback_for_llm() 결과
            remaining_primitives: 아직 실행하지 않은 primitive ids

        Returns:
            새로운 primitive id 리스트 (나머지 계획)
        """
        scene_primitives = self.get_primitives_for_task(task_name)
        if not scene_primitives:
            return remaining_primitives  # fallback: 기존 계획 유지

        task_description = " ".join(task_name.split("_")[2:])
        remaining_desc   = ", ".join(remaining_primitives)

        system_prompt = build_replan_system_prompt(scene_primitives)
        user_message  = (
            f'Task: "{task_description}"\n\n'
            f"Execution history so far:\n{memory_feedback}\n\n"
            f"Remaining planned steps (may need adjustment): [{remaining_desc}]\n\n"
            f"Based on what happened, what should the robot do next? "
            f"Provide the updated remaining plan as a JSON array."
        )

        log.info(f"[Replan] Calling LLM with feedback:\n{memory_feedback}")

        try:
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            log.info(f"[Replan] LLM response: {raw}")

            new_plan = json.loads(raw)
            valid_ids = {p["id"] for p in scene_primitives}
            new_plan  = [pid for pid in new_plan if pid in valid_ids]

            if new_plan:
                log.info(f"[Replan] New plan: {new_plan}")
                return new_plan
        except Exception as e:
            log.warning(f"[Replan] LLM call failed: {e}. Keeping original plan.")

        return remaining_primitives  # fallback

    def execute(
        self,
        env,
        vla_model,
        task_name: str,
        task_embs: dict,
        obs,
        record: bool = False,
    ) -> tuple[bool, int, list]:
        """
        Execute a LIBERO-10 task via LLM decomposition + sequential VLA calls.

        Returns:
            (success, total_steps, frames)  frames is [] if record=False
        """
        # Save home pose from initial observation
        home_eef_pos = obs["robot0_eef_pos"].copy()
        home_eef_quat = obs["robot0_eef_quat"].copy()
        home_gripper = obs["robot0_gripper_qpos"].mean()  # scalar

        # Decompose task
        primitive_ids = self.decompose(task_name)
        log.info(f"Plan for '{task_name}': {primitive_ids}")

        total_steps = 0
        success = False
        frames = []

        for step_idx, primitive_id in enumerate(primitive_ids):
            # Get embedding for this primitive
            if primitive_id not in task_embs:
                log.warning(f"No embedding for primitive '{primitive_id}'. Skipping.")
                continue

            prim_emb = task_embs[primitive_id].to(self.device).unsqueeze(0)
            log.info(f"[{step_idx+1}/{len(primitive_ids)}] Executing: {primitive_id}")

            # Reset VLA internal state for new primitive
            vla_model.reset()

            # Run VLA for steps_per_primitive frames
            for frame in range(self.steps_per_primitive):
                agentview = (
                    torch.from_numpy(obs["agentview_image"])
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                    / 255.0
                )
                eye = (
                    torch.from_numpy(obs["robot0_eye_in_hand_image"])
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                    / 255.0
                )
                robot_state = torch.from_numpy(
                    np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
                ).float().unsqueeze(0).unsqueeze(0).to(self.device)

                action = vla_model.predict(
                    {
                        "agentview_image": agentview,
                        "eye_in_hand_image": eye,
                        "lang_emb": prim_emb,
                        "robot_states": robot_state,
                    }
                ).cpu().numpy()

                obs, reward, done, _ = env.step(action)
                total_steps += 1

                if record:
                    frames.append(obs["agentview_image"].copy())

                if reward == 1:
                    success = True
                    log.info(f"SUCCESS at step {total_steps} (primitive {step_idx+1})")
                    return success, total_steps, frames

                if done:
                    break

            # Return to home pose between primitives (skip after last primitive)
            if step_idx < len(primitive_ids) - 1:
                log.info(f"Returning to home pose after primitive {step_idx+1}")
                obs, home_frames = go_to_home_pose(
                    env, obs, home_eef_pos, home_eef_quat, home_gripper, record=record
                )
                total_steps += HOME_STEPS
                if record:
                    frames.extend(home_frames)

        return success, total_steps, frames
