"""
IDEA2 v2 Evaluation with EpisodeMemory + Online Context.

각 primitive 전후 z_scene을 캡처해 delta 계산.
EpisodeMemory에 결과 기록 → 다음 primitive에 context_vec 제공.
실패 시 LLM에 피드백 전달 → replan.

Usage:
    python -m hierarchical_vla.idea2.eval_idea2 \\
        --checkpoint ./checkpoints/idea2_v2/stage3/final_model.pth \\
        --vqvae_path ./checkpoints/idea2_v2/stage1/vqvae.pt \\
        --cache_dir ./resnet_cache/libero_10_subset \\
        --num_rollouts 3
"""

import argparse
import logging
import os
import pickle

import imageio
import numpy as np
import torch

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from hierarchical_vla.libero_bench.llm_orchestrator import (
    LLMOrchestrator, go_to_home_pose, HOME_STEPS,
)
from .model_factory import create_idea2_model, create_context_encoder
from .memory import EpisodeMemory
from .vqvae import VQVAEModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    vqvae_path: str,
    cache_dir: str,
    device: str = "cuda",
    K: int = 64,
):
    scene_encoder_path = os.path.join(cache_dir, "scene_encoder.pt")
    model = create_idea2_model(
        K=K,
        device=device,
        vqvae_path=vqvae_path,
        scene_encoder_path=scene_encoder_path,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    scaler_path = os.path.join(os.path.dirname(checkpoint_path), "model_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            model.set_scaler(pickle.load(f))
    else:
        log.warning("Scaler not found.")

    model.eval()
    return model


def _frame_to_tensor(frame_np: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, C) uint8 → (1, C, H, W) float32 [0,1]"""
    return (
        torch.from_numpy(frame_np).float()
        .permute(2, 0, 1).unsqueeze(0) / 255.0
    ).to(device)


def execute_with_memory(
    orchestrator: LLMOrchestrator,
    model,
    context_encoder,
    memory: EpisodeMemory,
    env,
    task_name: str,
    task_embs: dict,
    obs,
    record: bool = False,
    max_replans: int = 2,
) -> tuple:
    """
    EpisodeMemory 기반 primitive 실행.
    실패 시 LLM에 피드백 전달 → replan (max_replans회까지).
    """
    device = model.device
    home_eef_pos  = obs["robot0_eef_pos"].copy()
    home_eef_quat = obs["robot0_eef_quat"].copy()
    home_gripper  = obs["robot0_gripper_qpos"].mean()

    total_steps = 0
    success     = False
    frames      = []

    # 초기 plan
    primitive_ids = orchestrator.decompose(task_name)
    log.info(f"Plan: {primitive_ids}")

    replan_count = 0
    step_idx     = 0
    episode_done = False  # 전체 episode 종료 flag

    while step_idx < len(primitive_ids) and not episode_done:
        primitive_id = primitive_ids[step_idx]

        if primitive_id not in task_embs:
            log.warning(f"No embedding for '{primitive_id}'. Skipping.")
            step_idx += 1
            continue

        prim_emb = task_embs[primitive_id].to(device).unsqueeze(0)
        log.info(f"[{step_idx+1}/{len(primitive_ids)}] {primitive_id}")

        # ── context_vec: memory → ContextEncoder ──────────────────────
        delta_history = memory.get_delta_history()
        context_vec   = context_encoder(delta_history).to(device)  # (scene_dim,)

        # ── z_before ──────────────────────────────────────────────────
        img_before = _frame_to_tensor(obs["agentview_image"], device)
        with torch.no_grad():
            z_before = model.scene_encoder(img_before).squeeze(0)  # (D,)

        model.reset()
        primitive_success = False

        # ── VLA 실행 ──────────────────────────────────────────────────
        for _ in range(orchestrator.steps_per_primitive):
            agentview = _frame_to_tensor(obs["agentview_image"], device).unsqueeze(0)
            eye       = _frame_to_tensor(obs["robot0_eye_in_hand_image"], device).unsqueeze(0)
            robot_state = torch.from_numpy(
                np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
            ).float().unsqueeze(0).unsqueeze(0).to(device)

            action, z_scene, k = model.predict(
                {
                    "agentview_image":   agentview,
                    "eye_in_hand_image": eye,
                    "lang_emb":          prim_emb,
                    "robot_states":      robot_state,
                },
                context_vec=context_vec,
            )

            obs, reward, done, _ = env.step(action.cpu().numpy())
            total_steps += 1

            if record:
                frames.append(obs["agentview_image"].copy())

            if reward == 1:
                primitive_success = True
                success = True
                log.info(f"SUCCESS at step {total_steps}")
                # 마지막 z_after로 memory 기록
                img_after = _frame_to_tensor(obs["agentview_image"], device)
                with torch.no_grad():
                    z_after = model.scene_encoder(img_after).squeeze(0)
                delta = z_after - z_before
                memory.write(primitive_id, delta, True, k)
                return success, total_steps, frames

            if done:
                episode_done = True
                break

        # ── z_after ───────────────────────────────────────────────────
        img_after = _frame_to_tensor(obs["agentview_image"], device)
        with torch.no_grad():
            z_after = model.scene_encoder(img_after).squeeze(0)

        delta = z_after - z_before

        # latent distance로 성공 보조 판단
        latent_ok = model.check_success_latent(delta)
        status = "OK" if primitive_success or latent_ok else "FAIL"
        log.info(
            f"  {primitive_id}: {status} | "
            f"delta_norm={delta.norm().item():.3f} | k={k}"
        )

        memory.write(primitive_id, delta, primitive_success or latent_ok, k)

        if episode_done:
            log.info("Episode terminated. Ending rollout.")
            break

        # ── 실패 시 replan ─────────────────────────────────────────────
        if not (primitive_success or latent_ok) and replan_count < max_replans:
            replan_count += 1
            feedback = memory.get_feedback_for_llm()
            log.info(f"Replanning ({replan_count}/{max_replans}):\n{feedback}")

            new_plan = orchestrator.replan(task_name, feedback, primitive_ids[step_idx:])
            if new_plan:
                log.info(f"New plan from step {step_idx}: {new_plan}")
                primitive_ids = primitive_ids[:step_idx] + new_plan
                continue

        # ── Home pose 복귀 ─────────────────────────────────────────────
        if step_idx < len(primitive_ids) - 1 and not episode_done:
            obs, home_frames = go_to_home_pose(
                env, obs, home_eef_pos, home_eef_quat, home_gripper, record=record
            )
            total_steps += HOME_STEPS
            if record:
                frames.extend(home_frames)

        step_idx += 1

    return success, total_steps, frames


def run_eval(
    checkpoint_path: str,
    vqvae_path: str,
    cache_dir: str,
    openai_api_key: str,
    K: int = 64,
    benchmark_type: str = "libero_10",
    num_rollouts: int = 3,
    steps_per_primitive: int = 250,
    device: str = "cuda",
    seed: int = 42,
    gpt_model: str = "gpt-4o",
    video_dir: str = None,
    max_replans: int = 2,
):
    PRIMITIVES_PATH = os.path.join(
        os.path.dirname(__file__), "..", "libero_bench", "libero_primitives.json"
    )
    LANG_EMB_DIR = os.path.join(
        os.path.dirname(__file__), "..", "language_embeddings"
    )

    log.info(f"Loading IDEA2 v2 model from {checkpoint_path}")
    model = load_model(
        checkpoint_path, vqvae_path, cache_dir, device=device, K=K
    )
    context_encoder = create_context_encoder(scene_dim=256)

    # Primitive embeddings
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

            # memory: rollout마다 초기화
            mem_path = None
            if video_dir:
                mem_path = os.path.join(
                    video_dir, f"memory_task{task_idx:02d}_roll{rollout_idx}.json"
                )
            memory = EpisodeMemory(path=mem_path)
            memory.reset()

            success, steps, frames = execute_with_memory(
                orchestrator=orchestrator,
                model=model,
                context_encoder=context_encoder,
                memory=memory,
                env=env,
                task_name=task_name,
                task_embs=prim_embs,
                obs=obs,
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
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--vqvae_path",      type=str, required=True)
    parser.add_argument("--cache_dir",       type=str, required=True)
    parser.add_argument("--openai_api_key",  type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--K",               type=int, default=64)
    parser.add_argument("--num_rollouts",    type=int, default=3)
    parser.add_argument("--steps_per_primitive", type=int, default=250)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--gpt_model",       type=str, default="gpt-4o")
    parser.add_argument("--video_dir",       type=str, default=None)
    parser.add_argument("--max_replans",     type=int, default=2)
    args = parser.parse_args()

    if not args.openai_api_key:
        raise ValueError("Provide --openai_api_key or set OPENAI_API_KEY env var")

    run_eval(
        checkpoint_path=args.checkpoint,
        vqvae_path=args.vqvae_path,
        cache_dir=args.cache_dir,
        openai_api_key=args.openai_api_key,
        K=args.K,
        num_rollouts=args.num_rollouts,
        steps_per_primitive=args.steps_per_primitive,
        device=args.device,
        seed=args.seed,
        gpt_model=args.gpt_model,
        video_dir=args.video_dir,
        max_replans=args.max_replans,
    )
