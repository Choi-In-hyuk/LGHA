"""
IDEA3 Evaluation: MambaVLA + obs_embed compensation.

핵심 아이디어:
  primitive 실행 전후의 obs_embed 차이(delta)를 EpisodeMemory에 저장.
  다음 primitive 실행 시 누적 delta를 obs_embed에서 빼서
  MambaVLA가 학습 분포에 가까운 표현을 받도록 보정.

  z_compensated = obs_embed_now - cumulative_delta ≈ obs_embed_train

재학습 불필요. 기존 MambaVLA 체크포인트 그대로 사용.

Usage:
    python -m hierarchical_vla.idea3.eval \\
        --checkpoint /home/choi/LGHA/hierarchical_vla/checkpoints/libero_10_subset/final_model.pth \\
        --openai_api_key $OPENAI_API_KEY \\
        --num_rollouts 3 \\
        --video_dir ./videos/idea3
"""

import argparse
import logging
import os
import pickle

import imageio
import numpy as np
import torch
import torch.nn as nn

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from hierarchical_vla.libero_bench.llm_orchestrator import (
    LLMOrchestrator, go_to_home_pose, HOME_STEPS,
)
from .memory import EpisodeMemory
from .vla_with_comp import MambaVLAWithComp
from .train import load_base_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

PRIMITIVES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "libero_bench", "libero_primitives.json"
)
LANG_EMB_DIR = os.path.join(
    os.path.dirname(__file__), "..", "language_embeddings"
)

# env obs key → model input key
ENV_TO_MODEL_KEY = {
    "agentview_image":           "agentview_image",
    "robot0_eye_in_hand_image":  "eye_in_hand_image",
}
CAM_NAMES = ["agentview", "eye_in_hand"]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cuda") -> MambaVLAWithComp:
    """MambaVLAWithComp 로드."""
    model = load_base_model(checkpoint_path, device)
    model.eval()
    log.info(f"Loaded MambaVLAWithComp from {checkpoint_path}")
    return model


# ── Obs embed helper ───────────────────────────────────────────────────────────

@torch.no_grad()
def get_obs_embed(model: MambaVLAWithComp, obs: dict, device: str) -> torch.Tensor:
    """
    env obs(numpy dict) → obs_embed (1, num_cams, latent_dim).
    comp_delta 없이 순수 이미지 인코딩만 한다.
    """
    img_dict = {}
    for env_key, model_key in ENV_TO_MODEL_KEY.items():
        img = obs[env_key]  # (H, W, C) uint8
        img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_dict[model_key] = img_t.to(device)

    # comp_delta 일시 해제 후 raw embed 계산
    saved = model._comp_delta
    model.set_comp_delta(None)
    obs_embed = model.img_encoder(img_dict)  # (1, num_cams, latent_dim)
    model.set_comp_delta(saved)
    return obs_embed


# ── Execution with compensation ────────────────────────────────────────────────

def execute_with_compensation(
    orchestrator: LLMOrchestrator,
    model,
    memory: EpisodeMemory,
    env,
    task_name: str,
    task_embs: dict,
    obs,
    device: str,
    record: bool = False,
    max_replans: int = 2,
) -> tuple:
    """
    LLM plan → primitives 순차 실행 (obs_embed compensation 적용).
    실패 시 replan.
    """
    home_eef_pos  = obs["robot0_eef_pos"].copy()
    home_eef_quat = obs["robot0_eef_quat"].copy()
    home_gripper  = obs["robot0_gripper_qpos"].mean()

    total_steps = 0
    success     = False
    frames      = []

    primitive_ids = orchestrator.decompose(task_name)
    log.info(f"Plan: {primitive_ids}")

    replan_count = 0
    step_idx     = 0

    while step_idx < len(primitive_ids):
        primitive_id = primitive_ids[step_idx]

        if primitive_id not in task_embs:
            log.warning(f"No embedding for '{primitive_id}'. Skipping.")
            step_idx += 1
            continue

        prim_emb = task_embs[primitive_id].to(device).unsqueeze(0)
        log.info(f"[{step_idx+1}/{len(primitive_ids)}] {primitive_id}")

        # ── 보정 설정 ────────────────────────────────────────────────────────
        cumulative_delta = memory.get_cumulative_delta()
        model.set_comp_delta(cumulative_delta)

        # ── obs_embed_before ─────────────────────────────────────────────────
        obs_embed_before = get_obs_embed(model, obs, device)  # (1, num_cams, D)

        model.reset()
        primitive_success = False

        # ── VLA 실행 ─────────────────────────────────────────────────────────
        for _ in range(orchestrator.steps_per_primitive):
            agentview = (
                torch.from_numpy(obs["agentview_image"])
                .float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            )
            eye = (
                torch.from_numpy(obs["robot0_eye_in_hand_image"])
                .float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            )
            robot_state = torch.from_numpy(
                np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
            ).float().unsqueeze(0).unsqueeze(0).to(device)

            action = model.predict({
                "agentview_image":   agentview,
                "eye_in_hand_image": eye,
                "lang_emb":          prim_emb,
                "robot_states":      robot_state,
            }).cpu().numpy()

            obs, reward, done, _ = env.step(action)
            total_steps += 1

            if record:
                frames.append(obs["agentview_image"].copy())

            if reward == 1:
                primitive_success = True
                success = True
                log.info(f"SUCCESS at step {total_steps}")
                # delta 기록 후 종료
                obs_embed_after = get_obs_embed(model, obs, device)
                delta = (obs_embed_after - obs_embed_before).squeeze(0)  # (num_cams, D)
                memory.write(primitive_id, delta, True)
                return success, total_steps, frames

            if done:
                break

        # ── obs_embed_after & delta 기록 ─────────────────────────────────────
        obs_embed_after = get_obs_embed(model, obs, device)
        delta = (obs_embed_after - obs_embed_before).squeeze(0)  # (num_cams, D)

        delta_norm = delta.norm().item()
        log.info(
            f"  {primitive_id}: {'OK' if primitive_success else 'FAIL'} "
            f"| delta_norm={delta_norm:.3f}"
        )
        memory.write(primitive_id, delta, primitive_success)

        # ── 실패 시 replan ───────────────────────────────────────────────────
        if not primitive_success and replan_count < max_replans:
            replan_count += 1
            feedback = memory.get_feedback_for_llm()
            log.info(f"Replanning ({replan_count}/{max_replans}):\n{feedback}")
            new_plan = orchestrator.replan(task_name, feedback, primitive_ids[step_idx:])
            if new_plan:
                log.info(f"New plan from step {step_idx}: {new_plan}")
                primitive_ids = primitive_ids[:step_idx] + new_plan
                continue

        # ── Home pose ────────────────────────────────────────────────────────
        if step_idx < len(primitive_ids) - 1:
            obs, home_frames = go_to_home_pose(
                env, obs, home_eef_pos, home_eef_quat, home_gripper, record=record
            )
            total_steps += HOME_STEPS
            if record:
                frames.extend(home_frames)

        step_idx += 1

    return success, total_steps, frames


# ── Main eval loop ─────────────────────────────────────────────────────────────

def run_eval(
    checkpoint_path: str,
    openai_api_key: str,
    benchmark_type: str = "libero_10",
    num_rollouts: int = 3,
    steps_per_primitive: int = 250,
    device: str = "cuda",
    seed: int = 42,
    gpt_model: str = "gpt-4o",
    video_dir: str = None,
    max_replans: int = 2,
):
    model = load_model(checkpoint_path, device)

    emb_path = os.path.join(LANG_EMB_DIR, "libero_90.pkl")
    with open(emb_path, "rb") as f:
        raw = pickle.load(f)
    prim_embs = {
        k: (v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))
        for k, v in raw.items()
    }
    log.info(f"Loaded {len(prim_embs)} primitive embeddings")

    orchestrator = LLMOrchestrator(
        primitives_path=PRIMITIVES_PATH,
        openai_api_key=openai_api_key,
        steps_per_primitive=steps_per_primitive,
        gpt_model=gpt_model,
        device=device,
    )

    bench     = benchmark.get_benchmark_dict()[benchmark_type]()
    num_tasks = 10

    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    results = {}

    for task_idx in range(num_tasks):
        task_bddl   = bench.get_task_bddl_file_path(task_idx)
        task_name   = os.path.basename(task_bddl).split(".")[0]
        init_states = bench.get_task_init_states(task_idx)

        task_successes = []

        for rollout_idx in range(num_rollouts):
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl,
                camera_heights=128,
                camera_widths=128,
                horizon=2000,
            )
            env.seed(seed + rollout_idx)
            env.reset()
            obs = env.set_init_state(init_states[rollout_idx % len(init_states)])

            dummy = np.zeros(7)
            dummy[-1] = -1.0
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            mem_path = None
            if video_dir:
                mem_path = os.path.join(
                    video_dir, f"memory_task{task_idx:02d}_roll{rollout_idx}.json"
                )
            memory = EpisodeMemory(path=mem_path)
            memory.reset()

            success, steps, frames = execute_with_compensation(
                orchestrator=orchestrator,
                model=model,
                memory=memory,
                env=env,
                task_name=task_name,
                task_embs=prim_embs,
                obs=obs,
                device=device,
                record=bool(video_dir),
                max_replans=max_replans,
            )

            task_successes.append(int(success))
            status = "SUCCESS" if success else "FAILED"
            log.info(
                f"[Task {task_idx:02d} | Roll {rollout_idx}] "
                f"{task_name} → {status} ({steps} steps)"
            )

            if video_dir and frames:
                video_path = os.path.join(
                    video_dir, f"task{task_idx:02d}_roll{rollout_idx}_{status}.mp4"
                )
                imageio.mimsave(video_path, [f[::-1] for f in frames], fps=20)

            env.close()

        rate = np.mean(task_successes)
        results[task_name] = {"success_rate": rate, "successes": task_successes}
        log.info(f"Task {task_idx:02d}: {rate*100:.1f}% ({sum(task_successes)}/{num_rollouts})")

    avg = np.mean([v["success_rate"] for v in results.values()])
    log.info("=" * 60)
    log.info(f"OVERALL: {avg*100:.1f}%")
    log.info("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",          type=str, required=True)
    parser.add_argument("--openai_api_key",      type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--benchmark_type",      type=str, default="libero_10")
    parser.add_argument("--num_rollouts",        type=int, default=3)
    parser.add_argument("--steps_per_primitive", type=int, default=250)
    parser.add_argument("--device",              type=str, default="cuda")
    parser.add_argument("--seed",                type=int, default=42)
    parser.add_argument("--gpt_model",           type=str, default="gpt-4o")
    parser.add_argument("--video_dir",           type=str, default=None)
    parser.add_argument("--max_replans",         type=int, default=2)
    args = parser.parse_args()

    if not args.openai_api_key:
        raise ValueError("Provide --openai_api_key or set OPENAI_API_KEY env var")

    run_eval(
        checkpoint_path=args.checkpoint,
        openai_api_key=args.openai_api_key,
        benchmark_type=args.benchmark_type,
        num_rollouts=args.num_rollouts,
        steps_per_primitive=args.steps_per_primitive,
        device=args.device,
        seed=args.seed,
        gpt_model=args.gpt_model,
        video_dir=args.video_dir,
        max_replans=args.max_replans,
    )
