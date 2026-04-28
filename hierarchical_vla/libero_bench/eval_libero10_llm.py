"""
LIBERO-10 evaluation with LLM orchestration.

GPT-4o decomposes each multi-step task into LIBERO-90 primitives.
MambaVLA executes each primitive sequentially with home-pose return.

Usage:
    python -m libero_bench.eval_libero10_llm \
        --checkpoint /home/choi/MambaVLA/checkpoints/libero_object/final_model.pth \
        --openai_api_key sk-... \
        --num_rollouts 3
"""

import argparse
import logging
import os
import pickle
import numpy as np
import torch
import imageio

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from MambaVLA.model_factory import create_mambavla_model

from .llm_orchestrator import LLMOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

PRIMITIVES_PATH = os.path.join(os.path.dirname(__file__), "libero_primitives.json")
LANG_EMB_DIR = os.path.join(os.path.dirname(__file__), "..", "language_embeddings")


def load_model(checkpoint_path: str, device: str = "cuda"):
    camera_names = ["agentview", "eye_in_hand"]
    model = create_mambavla_model(
        camera_names=camera_names,
        latent_dim=256,
        action_dim=7,
        lang_emb_dim=512,
        embed_dim=256,
        obs_tok_len=len(camera_names),
        action_seq_len=10,
        device=device,
        n_layer=5,
        d_intermediate=256,
        state_dim=45,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    # Load scaler
    scaler_path = os.path.join(os.path.dirname(checkpoint_path), "model_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            model.set_scaler(pickle.load(f))
    else:
        log.warning(f"Scaler not found at {scaler_path}. Predictions may be unscaled.")

    model.eval()
    return model


def load_task_embeddings(benchmark_type: str) -> dict:
    emb_path = os.path.join(LANG_EMB_DIR, f"{benchmark_type}.pkl")
    with open(emb_path, "rb") as f:
        raw = pickle.load(f)
    return {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))
            for k, v in raw.items()}


def run_eval(
    checkpoint_path: str,
    openai_api_key: str,
    benchmark_type: str = "libero_10",
    num_rollouts: int = 3,
    steps_per_primitive: int = 150,
    max_steps_total: int = 600,
    device: str = "cuda",
    seed: int = 42,
    gpt_model: str = "gpt-4o",
    video_dir: str = None,
):
    # Load model
    log.info(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # Load task embeddings for all LIBERO-90 primitives (used as sub-commands)
    prim_embs = load_task_embeddings("libero_90")
    log.info(f"Loaded {len(prim_embs)} primitive embeddings")

    # Setup orchestrator
    orchestrator = LLMOrchestrator(
        primitives_path=PRIMITIVES_PATH,
        openai_api_key=openai_api_key,
        steps_per_primitive=steps_per_primitive,
        gpt_model=gpt_model,
        device=device,
    )

    # Setup LIBERO benchmark
    bench = benchmark.get_benchmark_dict()[benchmark_type]()
    num_tasks = 10
    log.info(f"Evaluating on {num_tasks} LIBERO-10 tasks, {num_rollouts} rollouts each")

    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    results = {}

    for task_idx in range(num_tasks):
        task_bddl = bench.get_task_bddl_file_path(task_idx)
        task_name = os.path.basename(task_bddl).split(".")[0]
        init_states = bench.get_task_init_states(task_idx)

        task_successes = []

        for rollout_idx in range(num_rollouts):
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl,
                camera_heights=128,
                camera_widths=128,
            )
            env.seed(seed + rollout_idx)
            env.reset()
            obs = env.set_init_state(init_states[rollout_idx % len(init_states)])

            # Stabilize environment with dummy actions
            dummy = np.zeros(7)
            dummy[-1] = -1.0
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            model.reset()

            success, steps, frames = orchestrator.execute(
                env=env,
                vla_model=model,
                task_name=task_name,
                task_embs=prim_embs,
                obs=obs,
                record=bool(video_dir),
            )

            task_successes.append(int(success))
            status = "SUCCESS" if success else "FAILED"
            log.info(f"[Task {task_idx:02d} | Roll {rollout_idx}] {task_name} -> {status} ({steps} steps)")

            if video_dir and frames:
                video_path = os.path.join(
                    video_dir, f"task{task_idx:02d}_roll{rollout_idx}_{status}.mp4"
                )
                imageio.mimsave(video_path, [f[::-1] for f in frames], fps=20)
                log.info(f"Saved video: {video_path}")

            env.close()

        rate = np.mean(task_successes)
        results[task_name] = {"success_rate": rate, "successes": task_successes}
        log.info(f"Task {task_idx:02d} success rate: {rate*100:.1f}% ({sum(task_successes)}/{num_rollouts})")

    # Summary
    avg = np.mean([v["success_rate"] for v in results.values()])
    log.info("=" * 60)
    log.info(f"OVERALL SUCCESS RATE: {avg*100:.1f}%")
    log.info("=" * 60)
    for task_name, res in results.items():
        log.info(f"  {task_name}: {res['success_rate']*100:.1f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--openai_api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--benchmark_type", type=str, default="libero_10")
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--steps_per_primitive", type=int, default=250,
                        help="Max VLA steps per sub-command before home pose return")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Directory to save rollout videos (mp4). None = no recording.")
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
    )
