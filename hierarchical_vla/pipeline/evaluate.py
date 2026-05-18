"""
LGHA 평가 스크립트.

LIBERO-Object 벤치마크에서 계층 구조 모델 (Qwen2-VL + MambaVLA) 성능을 측정한다.

사용법:
  python -m hierarchical_vla.pipeline.evaluate \
      --checkpoint /home/choi/LGHA/checkpoints/hierarchical/final.pth \
      --suite libero_object \
      --num_episodes 20 \
      --device cuda

흐름 (매 스텝):
  1. 시뮬레이터 → 현재 이미지
  2. [최초 1회] Qwen2-VL → action_type (task 분류만)
     [매 스텝]  GT 투영    → loc_token  (시뮬레이터 물체 위치 → 정확한 좌표)
  3. MambaVLA → robot action (7DOF)
  4. action → 시뮬레이터 실행

개선 사항 vs. 이전 버전:
  - loc_token: Qwen2-VL zero-shot → 시뮬레이터 GT 투영 (물체 위치 변화에도 정확)
  - Qwen2-VL 부하 감소: episode당 1회 호출 (이전: 매 스텝)
"""

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
from io import BytesIO

import numpy as np
import torch
import torch.distributed as dist
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "MambaVLA"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LIBERO"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_MAX_STEPS    = 600   # episode당 최대 스텝 수
_GRASP_TIMEOUT = 100  # GRASP phase 연속 유지 스텝 수 초과 시 실패 판정

from .gt_localizer import (get_gt_loc_token, get_gt_target_loc_token,
                           PhaseDetector, PHASE_NAMES, NUM_PHASES,
                           REACH, GRASP, TRANSPORT, PLACE, RELEASE,
                           extract_object_name, get_body_pos)


def run_episode(
    env, task, mamba_model, orchestrator, device: str,
    video_path: str = None, debug: bool = False,
    use_phase: bool = False,
    use_qwen_lora: bool = False,
    qwen_stride: int = 5,
    corrected_loc_token: int = None,      # retry 시 REACH/GRASP 단계에 강제 적용
    failure_data_path: str = None,        # grasp 실패 JSONL 저장 경로
    qwen_failure_data_path: str = None,   # 실패 프레임을 Qwen 학습 포맷으로 저장
    grasp_timeout: int = _GRASP_TIMEOUT,
    early_exit_on_grasp_fail: bool = False,  # True면 첫 실패 시 즉시 종료
) -> tuple:
    """
    단일 episode 실행. (success: bool, grasp_failures: list) 반환.

    grasp_failures: 각 원소는 {"eef_pos", "obj_pos", "loc_token_used",
                                "gt_loc_token", "phase", "success"} dict.

    use_qwen_lora=True:
      매 qwen_stride 스텝마다 파인튜닝된 Qwen2-VL로 obj_loc + target_loc 추론.

    use_qwen_lora=False (기본):
      GT 투영으로 loc_token 계산.
    """
    obs = env.reset()
    mamba_model.reset()

    instruction = task.language
    task_name   = task.name
    writer      = None
    grasp_failures = []
    grasp_start_step = None   # GRASP phase 진입 스텝 (timeout 감지용)

    if use_phase:
        phase_detector = PhaseDetector(env, task_name, obs)
        action_type_id = 0
        log.info(f"  [{task_name}] phase 모드 ({'Qwen LoRA' if use_qwen_lora else 'GT'})")
    else:
        agentview0 = obs["agentview_image"]
        action_type_id, _ = orchestrator.infer(agentview0, instruction)
        log.info(f"  [{task_name}] action_type={action_type_id} (from Qwen2-VL)")

    _cached_obj_loc    = 512
    _cached_target_loc = 512

    def to_tensor(img):
        return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    def _close_writer():
        if writer:
            writer.release()
            _mp4_to_gif(video_path)

    for step in range(_MAX_STEPS):
        agentview   = obs["agentview_image"]
        eye_in_hand = obs["robot0_eye_in_hand_image"]

        if use_phase:
            action_type_id = phase_detector.step(obs)

        # loc_token 계산
        if use_qwen_lora:
            if step % qwen_stride == 0:
                _, _cached_obj_loc, _cached_target_loc = orchestrator.infer_dual_loc(
                    agentview, instruction
                )
            loc_token = (_cached_target_loc
                         if use_phase and action_type_id in (TRANSPORT, PLACE, RELEASE)
                         else _cached_obj_loc)
        elif use_phase and action_type_id in (TRANSPORT, PLACE, RELEASE):
            loc_token = get_gt_target_loc_token(env, task_name,
                                                camera="agentview", img_h=128, img_w=128)
        else:
            loc_token = get_gt_loc_token(env, task_name,
                                         camera="agentview", img_h=128, img_w=128)

        # retry 시 REACH/GRASP 구간에서 보정된 loc_token 강제 적용
        if corrected_loc_token is not None and use_phase and action_type_id in (REACH, GRASP):
            loc_token = corrected_loc_token

        # ── grasp 실패 감지 (phase 모드 전용) ───────────────────────────────
        if use_phase:
            if action_type_id == GRASP:
                if grasp_start_step is None:
                    grasp_start_step = step
                elif step - grasp_start_step >= grasp_timeout:
                    eef_pos = obs.get("robot0_eef_pos")
                    obj_pos = get_body_pos(env.sim, extract_object_name(task_name))
                    if eef_pos is not None and obj_pos is not None:
                        gt_loc = get_gt_loc_token(env, task_name,
                                                  camera="agentview", img_h=128, img_w=128)
                        record = {
                            "eef_pos":        eef_pos.tolist(),
                            "obj_pos":        obj_pos.tolist(),
                            "loc_token_used": int(loc_token),
                            "gt_loc_token":   int(gt_loc),
                            "phase":          "grasp",
                            "success":        False,
                        }
                        grasp_failures.append(record)
                        if failure_data_path:
                            with open(failure_data_path, "a") as f:
                                f.write(json.dumps(record) + "\n")
                        log.info(f"  Grasp failure at step={step} | loc_used={loc_token} gt={gt_loc}")

                        # Qwen 재학습용: 실패 프레임 + GT loc_token을 학습 포맷으로 저장
                        if qwen_failure_data_path is not None:
                            from PIL import Image as PILImage
                            target_loc = get_gt_target_loc_token(env, task_name,
                                                                 camera="agentview", img_h=128, img_w=128)
                            pil = PILImage.fromarray(agentview.astype(np.uint8))
                            buf = BytesIO()
                            pil.save(buf, format="PNG")
                            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                            qwen_rec = {
                                "image":       img_b64,
                                "instruction": instruction,
                                "response":    f"object: {gt_loc} target: {target_loc}",
                                "obj_loc":     int(gt_loc),
                                "target_loc":  int(target_loc),
                                "task_name":   task_name,
                                "source":      "failure",
                            }
                            with open(qwen_failure_data_path, "a") as f:
                                f.write(json.dumps(qwen_rec) + "\n")
                    grasp_start_step = None   # 다음 GRASP 구간 감지를 위해 리셋
                    if early_exit_on_grasp_fail:
                        _close_writer()
                        return False, grasp_failures
            else:
                grasp_start_step = None

        # MambaVLA obs 구성 [1, 1, C, H, W]
        robot_state = np.concatenate([
            obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]
        ])
        obs_dict = {
            "agentview_image":   to_tensor(agentview),
            "eye_in_hand_image": to_tensor(eye_in_hand),
            "robot_states":      torch.from_numpy(robot_state).float().unsqueeze(0).unsqueeze(0).to(device),
        }

        action = mamba_model.predict(
            obs_dict=obs_dict,
            loc_token=torch.tensor([loc_token], device=device),
            action_type=torch.tensor([action_type_id], device=device),
        )
        action = action.cpu().numpy()

        if debug and step % 10 == 0:
            obj_name = extract_object_name(task_name)
            obj_pos  = get_body_pos(env.sim, obj_name)
            obj_z    = round(float(obj_pos[2]), 3) if obj_pos is not None else "?"
            eef_z    = round(float(obs["robot0_eef_pos"][2]), 3) if "robot0_eef_pos" in obs else "?"
            grip     = round(float(np.abs(obs["robot0_gripper_qpos"]).sum()), 4)
            phase_str = PHASE_NAMES.get(action_type_id, str(action_type_id)) if use_phase else str(action_type_id)
            log.info(f"  step={step:3d} | {phase_str} | loc={loc_token} | obj_z={obj_z} eef_z={eef_z} grip={grip} | act={action.round(3)}")

        if video_path is not None:
            if writer is None:
                h, w = agentview.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
            frame = cv2.cvtColor(agentview, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 0)
            row, col = loc_token // 32, loc_token % 32
            cx = int(col / 31 * (w - 1))
            cy = (h - 1) - int(row / 31 * (h - 1))
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            writer.write(frame)

        obs, reward, done, info = env.step(action)

        if done:
            _close_writer()
            success = bool(reward) or info.get("success", False)
            log.info(f"  done at step={step} | reward={reward} | success={success}")
            return success, grasp_failures

    _close_writer()
    return False, grasp_failures


def _mp4_to_gif(mp4_path: str):
    """ffmpeg로 MP4 → GIF 변환 (팔레트 최적화)."""
    if mp4_path is None or not os.path.exists(mp4_path):
        return
    gif_path = mp4_path.replace(".mp4", ".gif")
    palette  = mp4_path.replace(".mp4", "_palette.png")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", mp4_path,
            "-vf", "fps=15,scale=256:-1:flags=lanczos,palettegen",
            palette
        ], check=True, capture_output=True)
        subprocess.run([
            "ffmpeg", "-y", "-i", mp4_path, "-i", palette,
            "-lavfi", "fps=15,scale=256:-1:flags=lanczos[x];[x][1:v]paletteuse",
            gif_path
        ], check=True, capture_output=True)
        os.remove(palette)
        log.info(f"  GIF saved → {gif_path}")
    except Exception as e:
        log.warning(f"  GIF 변환 실패: {e}")


def evaluate(
    checkpoint_path: str,
    suite_name: str = "libero_object",
    num_episodes: int = 20,
    device: str = "cuda:0",
    qwen_device: str = "cuda:1",
    qwen_model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    qwen_lora_path: str = None,
    seed: int = 42,
    save_video: bool = True,
    video_dir: str = "./eval_videos",
    debug: bool = False,
    use_phase: bool = False,
    qwen_stride: int = 5,
    failure_data_path: str = None,
    qwen_failure_data_path: str = None,
    correction_module_path: str = None,
    grasp_timeout: int = _GRASP_TIMEOUT,
):
    # ── DDP 초기화 ────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank >= 0

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device     = f"cuda:{local_rank}"
        qwen_device = f"cuda:{local_rank}"   # 각 rank가 자신의 GPU에 둘 다 로드
        world_size = dist.get_world_size()
    else:
        world_size = 1
        local_rank = 0

    is_main = (local_rank == 0)

    # ── 1. LIBERO 환경 로드 ─────────────────────────────────────────────────
    import torch as _torch
    _orig_load = _torch.load
    _torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bm_dict = benchmark.get_benchmark_dict()
    bm = bm_dict[suite_name]()
    num_tasks = bm.get_num_tasks()

    # 각 rank가 처리할 task 분배 (stride 방식: rank0=0,2,4,.. rank1=1,3,5,..)
    my_task_ids = list(range(local_rank, num_tasks, world_size))
    if is_main:
        log.info(f"Suite: {suite_name} | {num_tasks} tasks | {num_episodes} eps/task "
                 f"| DDP={'ON world_size=' + str(world_size) if is_ddp else 'OFF'}")
    log.info(f"  [rank{local_rank}] tasks: {my_task_ids} | device: {device}")

    # ── 2. MambaVLA 로드 ───────────────────────────────────────────────────
    from MambaVLA.model_factory import create_mambavla_model
    from hierarchical_vla.pipeline.action_type_mapper import NUM_ACTION_TYPES

    num_types = NUM_PHASES if use_phase else NUM_ACTION_TYPES
    model = create_mambavla_model(
        camera_names=["agentview", "eye_in_hand"],
        latent_dim=256, embed_dim=256,
        action_seq_len=10, n_layer=5,
        device=device,
        num_action_types=num_types,
        layer3_channels=256,
    )

    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_model_scaler(checkpoint_dir)
    # pickle로 저장된 scaler는 device가 고정돼 있어서 DDP 시 rank device로 수동 이동
    _scaler = model.scaler
    for attr in ("y_min", "y_max", "new_min_y", "new_max_y",
                 "y_bounds_tensor", "tensor_y_bounds"):
        if hasattr(_scaler, attr) and isinstance(getattr(_scaler, attr), torch.Tensor):
            setattr(_scaler, attr, getattr(_scaler, attr).to(device))
    if hasattr(_scaler, "device"):
        _scaler.device = device
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    log.info(f"  [rank{local_rank}] MambaVLA loaded on {device}")

    # ── 3. Qwen2-VL 로드 (같은 GPU) ────────────────────────────────────────
    from hierarchical_vla.pipeline.qwen_orchestrator import QwenOrchestrator
    orchestrator = QwenOrchestrator(
        model_id=qwen_model_id,
        device=qwen_device,
        lora_path=qwen_lora_path,
    )
    use_qwen_lora = qwen_lora_path is not None
    if use_qwen_lora and is_main:
        log.info(f"Qwen LoRA 모드: {qwen_lora_path} (stride={qwen_stride})")

    # ── 4. Correction Module 로드 ────────────────────────────────────────────
    correction_module = None
    if correction_module_path is not None:
        from hierarchical_vla.pipeline.correction_module import CorrectionModule
        correction_module = CorrectionModule.load(correction_module_path, device=device)

    # rank별 임시 failure 파일 경로
    def _rank_path(path):
        if path is None or not is_ddp:
            return path
        base, ext = os.path.splitext(path)
        return f"{base}.rank{local_rank}{ext}"

    my_failure_path      = _rank_path(failure_data_path)
    my_qwen_failure_path = _rank_path(qwen_failure_data_path)

    for p in [my_failure_path, my_qwen_failure_path]:
        if p:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # ── 5. 평가 루프 (이 rank에 할당된 task만) ────────────────────────────────
    total_success = 0
    total_episodes = 0
    results = {}

    for task_id in my_task_ids:
        task = bm.get_task(task_id)
        task_suite = bm.get_task_init_states(task_id)

        env_args = {
            "bddl_file_name": os.path.join(
                os.path.dirname(__file__), "..", "..", "LIBERO",
                "libero", "libero", "bddl_files",
                suite_name, f"{task.name}.bddl"
            ),
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "camera_heights": 128,
            "camera_widths": 128,
            "render_camera": "agentview",
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        task_success = 0
        if save_video:
            os.makedirs(os.path.join(video_dir, suite_name), exist_ok=True)

        for ep in range(num_episodes):
            init_state = task_suite[ep % len(task_suite)]
            env.set_init_state(init_state)
            video_path = None
            if save_video:
                video_path = os.path.join(video_dir, suite_name, f"{task.name}_ep{ep+1:02d}.mp4")

            success, grasp_failures = run_episode(
                env, task, model, orchestrator, device,
                video_path=video_path, debug=debug,
                use_phase=use_phase,
                use_qwen_lora=use_qwen_lora,
                qwen_stride=qwen_stride,
                failure_data_path=my_failure_path,
                qwen_failure_data_path=my_qwen_failure_path,
                grasp_timeout=grasp_timeout,
                early_exit_on_grasp_fail=(correction_module is not None),
            )

            if not success and grasp_failures and correction_module is not None:
                last = grasp_failures[-1]
                eef_pos   = np.array(last["eef_pos"])
                obj_pos   = np.array(last["obj_pos"])
                wrong_tok = last["loc_token_used"]
                corrected = correction_module.correct(eef_pos, obj_pos, wrong_tok, device=device)
                log.info(f"  [Correction] loc_token {wrong_tok} → {corrected} | retry ep={ep+1}")

                env.set_init_state(init_state)
                retry_video = video_path.replace(".mp4", "_retry.mp4") if video_path else None
                success, _ = run_episode(
                    env, task, model, orchestrator, device,
                    video_path=retry_video, debug=debug,
                    use_phase=use_phase,
                    use_qwen_lora=use_qwen_lora,
                    qwen_stride=qwen_stride,
                    corrected_loc_token=corrected,
                    grasp_timeout=grasp_timeout,
                )

            task_success += int(success)
            total_episodes += 1
            log.info(f"  [rank{local_rank}][{task.name}] ep {ep+1}/{num_episodes}: {'SUCCESS' if success else 'FAIL'}")

        env.close()
        rate = task_success / num_episodes
        results[task.name] = rate
        total_success += task_success
        log.info(f"  [rank{local_rank}] Task {task_id+1}/{num_tasks} | {task.name}: {rate*100:.1f}%")

    # ── 6. 분산 결과 집계 ────────────────────────────────────────────────────
    if is_ddp:
        t_succ = torch.tensor([total_success],  dtype=torch.long, device=device)
        t_eps  = torch.tensor([total_episodes], dtype=torch.long, device=device)
        dist.all_reduce(t_succ, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_eps,  op=dist.ReduceOp.SUM)
        total_success  = int(t_succ.item())
        total_episodes = int(t_eps.item())

        # rank 0가 임시 failure 파일들을 원본 경로로 합병
        dist.barrier()
        if is_main:
            for orig, paths in [
                (failure_data_path,      [_rank_path(failure_data_path)      if _rank_path(failure_data_path)      != failure_data_path      else None
                                          for r in range(world_size) for _rank_path in [lambda p, _r=r: (p.replace('.jsonl', f'.rank{_r}.jsonl') if p else None)]]),
                (qwen_failure_data_path, [_rank_path(qwen_failure_data_path) if _rank_path(qwen_failure_data_path) != qwen_failure_data_path else None
                                          for r in range(world_size) for _rank_path in [lambda p, _r=r: (p.replace('.jsonl', f'.rank{_r}.jsonl') if p else None)]]),
            ]:
                pass  # replaced below

            def _merge_rank_files(orig_path, n_ranks):
                if orig_path is None:
                    return
                base, ext = os.path.splitext(orig_path)
                with open(orig_path, "w") as out:
                    for r in range(n_ranks):
                        rpath = f"{base}.rank{r}{ext}"
                        if os.path.exists(rpath):
                            with open(rpath) as f:
                                out.write(f.read())
                            os.remove(rpath)

            _merge_rank_files(failure_data_path,      world_size)
            _merge_rank_files(qwen_failure_data_path, world_size)

    if is_main:
        overall = total_success / max(total_episodes, 1)
        log.info(f"\n{'='*60}")
        log.info(f"Overall success rate: {overall*100:.1f}% ({total_success}/{total_episodes})")
        log.info(f"{'='*60}")
        for name, rate in results.items():
            log.info(f"  {name}: {rate*100:.1f}%")

    if is_ddp:
        dist.destroy_process_group()

    overall = total_success / max(total_episodes, 1)
    return overall, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--suite",        default="libero_object")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--qwen_device",  default="cuda:1")
    parser.add_argument("--qwen_model",   default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--video_dir",    default="./eval_videos")
    parser.add_argument("--no_video",     action="store_true")
    parser.add_argument("--debug",        action="store_true")
    parser.add_argument("--phase",        action="store_true",
                        help="phase 모드 (재학습된 체크포인트 사용 시)")
    parser.add_argument("--qwen_lora",             default=None,
                        help="파인튜닝된 Qwen2-VL LoRA 경로 (None이면 GT 투영 사용)")
    parser.add_argument("--qwen_stride",           type=int, default=5,
                        help="Qwen2-VL 추론 간격 (스텝 수)")
    parser.add_argument("--failure_data",          default=None,
                        help="grasp 실패 데이터 저장 경로 (JSONL)")
    parser.add_argument("--qwen_failure_data",     default=None,
                        help="실패 프레임 Qwen 학습 포맷 저장 경로 (JSONL)")
    parser.add_argument("--correction_module",     default=None,
                        help="CorrectionModule 가중치 경로 (Stage 2: retry 활성화)")
    parser.add_argument("--grasp_timeout",         type=int, default=_GRASP_TIMEOUT,
                        help="GRASP phase timeout 스텝 수 (기본: 100)")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        suite_name=args.suite,
        num_episodes=args.num_episodes,
        device=args.device,
        qwen_device=args.qwen_device,
        qwen_model_id=args.qwen_model,
        qwen_lora_path=args.qwen_lora,
        seed=args.seed,
        save_video=not args.no_video,
        video_dir=args.video_dir,
        debug=args.debug,
        use_phase=args.phase,
        qwen_stride=args.qwen_stride,
        failure_data_path=args.failure_data,
        qwen_failure_data_path=args.qwen_failure_data,
        correction_module_path=args.correction_module,
        grasp_timeout=args.grasp_timeout,
    )


if __name__ == "__main__":
    main()
