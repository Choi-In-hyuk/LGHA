"""
Home pose controller 테스트 (LLM API 없음).

VLA로 N 스텝 실행 후 go_to_home_pose 호출, 영상 저장.

Usage:
    python -m libero_bench.test_home_pose \
        --checkpoint ./checkpoints/libero_10_subset_scene2/final_model.pth \
        --task_idx 0 \
        --vla_steps 80 \
        --video_path ./videos/home_pose_test.mp4
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

from hierarchical_vla.models.model_factory_scene import create_mambavla_scene_model
from .llm_orchestrator import go_to_home_pose, HOME_STEPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LANG_EMB_DIR = os.path.join(os.path.dirname(__file__), "..", "language_embeddings")


def load_model(checkpoint_path: str, device: str = "cuda"):
    model = create_mambavla_scene_model(
        camera_names=["agentview", "eye_in_hand"],
        latent_dim=256,
        action_dim=7,
        lang_emb_dim=512,
        scene_emb_dim=256,
        embed_dim=256,
        obs_tok_len=2,
        action_seq_len=10,
        device=device,
        n_layer=5,
        d_intermediate=256,
        state_dim=45,
        vjepa_n_finetune_layers=0,
        vjepa_n_frames=1,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    scaler_path = os.path.join(os.path.dirname(checkpoint_path), "model_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            model.set_scaler(pickle.load(f))
    model.eval()
    return model


def run_test(
    checkpoint_path: str,
    task_idx: int = 0,
    vla_steps: int = 80,
    video_path: str = "./videos/home_pose_test.mp4",
    device: str = "cuda",
    seed: int = 42,
):
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────
    log.info("Loading model...")
    model = load_model(checkpoint_path, device)

    # ── Task embedding (LIBERO-10 첫 번째 task 사용) ───────────────────────
    emb_path = os.path.join(LANG_EMB_DIR, "libero_10.pkl")
    with open(emb_path, "rb") as f:
        raw = pickle.load(f)
    task_keys = list(raw.keys())
    task_key  = task_keys[task_idx % len(task_keys)]
    lang_emb_np = raw[task_key]
    lang_emb = (
        torch.tensor(lang_emb_np, dtype=torch.float32)
        .unsqueeze(0).to(device)
    )
    log.info(f"Task: {task_key}")

    # ── Environment ───────────────────────────────────────────────────────
    bench = benchmark.get_benchmark_dict()["libero_10"]()
    task_bddl   = bench.get_task_bddl_file_path(task_idx)
    init_states = bench.get_task_init_states(task_idx)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_states[0])

    dummy = np.zeros(7)
    dummy[-1] = -1.0
    for _ in range(5):
        obs, _, _, _ = env.step(dummy)

    # Home pose 저장
    home_eef_pos  = obs["robot0_eef_pos"].copy()
    home_eef_quat = obs["robot0_eef_quat"].copy()
    home_gripper  = obs["robot0_gripper_qpos"].mean()

    frames = []
    model.reset()

    # ── Phase 1: VLA 실행 ─────────────────────────────────────────────────
    log.info(f"Phase 1: VLA execution ({vla_steps} steps)...")
    for step in range(vla_steps):
        agentview = (
            torch.from_numpy(obs["agentview_image"]).float()
            .permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0
        )
        eye = (
            torch.from_numpy(obs["robot0_eye_in_hand_image"]).float()
            .permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0
        )
        robot_state = torch.from_numpy(
            np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
        ).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            action = model.predict({
                "agentview_image":   agentview,
                "eye_in_hand_image": eye,
                "lang_emb":          lang_emb,
                "robot_states":      robot_state,
            }).cpu().numpy()

        obs, reward, done, _ = env.step(action)
        frames.append(obs["agentview_image"].copy())

        if done or reward == 1:
            log.info(f"Episode ended at step {step+1} (reward={reward})")
            break

    log.info(f"VLA done. Current eef_pos: {obs['robot0_eef_pos'].round(3)}")
    log.info(f"Home eef_pos:              {home_eef_pos.round(3)}")

    # ── Phase 2: Home pose return ─────────────────────────────────────────
    log.info(f"Phase 2: Returning to home pose ({HOME_STEPS} steps)...")
    before_home = obs["robot0_eef_pos"].copy()
    obs, home_frames = go_to_home_pose(
        env, obs, home_eef_pos, home_eef_quat, home_gripper, record=True
    )
    frames.extend(home_frames)

    after_home = obs["robot0_eef_pos"]
    pos_error = np.linalg.norm(after_home - home_eef_pos)
    log.info(f"Before home: {before_home.round(3)}")
    log.info(f"After home:  {after_home.round(3)}")
    log.info(f"Position error: {pos_error:.4f} m")

    env.close()

    # ── Save video ────────────────────────────────────────────────────────
    imageio.mimsave(video_path, [f[::-1] for f in frames], fps=20)
    log.info(f"Video saved: {video_path} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--task_idx",    type=int, default=0)
    parser.add_argument("--vla_steps",   type=int, default=80)
    parser.add_argument("--video_path",  type=str, default="./videos/home_pose_test.mp4")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    run_test(
        checkpoint_path=args.checkpoint,
        task_idx=args.task_idx,
        vla_steps=args.vla_steps,
        video_path=args.video_path,
        device=args.device,
        seed=args.seed,
    )
