import logging
import os
import sys
import cv2
import random
import numpy as np
import torch
import wandb
import hydra
import multiprocessing as mp
import warnings

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from tqdm import tqdm
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

log = logging.getLogger(__name__)


def log_eval(msg):
    # Print evaluation logging
    log.info(msg)


def safe_display_image(img, window_name, render_enabled):
    if not render_enabled:
        return
    try:
        if "DISPLAY" not in os.environ or os.environ.get("DISPLAY") == "":
            return
        cv2.imshow(window_name, img)
        cv2.waitKey(1)
    except Exception:
        pass


def safe_destroy_window(window_name, render_enabled):
    if not render_enabled:
        return
    try:
        if "DISPLAY" in os.environ and os.environ.get("DISPLAY") != "":
            cv2.destroyWindow(window_name)
    except Exception:
        pass


class MultiTaskSim:
    def __init__(
        self,
        rollouts,
        max_step_per_episode,
        benchmark_type,
        use_eye_in_hand,
        seed,
        device,
        render_image,
        n_cores,
        use_multiprocessing=True,
        save_video=False,
        save_video_dir=None,
    ):
        self.seed = seed
        self.device = device
        self.render_image = render_image
        self.n_cores = n_cores
        self.benchmark_type = benchmark_type
        self.use_eye_in_hand = use_eye_in_hand
        self.save_video = save_video
        self.save_video_dir = save_video_dir
        self.rollouts = rollouts
        self.max_step_per_episode = max_step_per_episode
        self.use_multiprocessing = use_multiprocessing

    def get_task_embs(self, task_embs):
        self.task_embs = task_embs

    def eval_model(
        self,
        contexts,
        context_ind,
        success,
        episode_lengths,
        pid,
        cpu_set,
        counter,
        all_runs,
        model=None,
        model_config=None,
        model_states=None,
    ):
        # Model init
        if model_config is not None:
            model = hydra.utils.instantiate(model_config)
            model.recover_model_state(
                model_states["model"], model_states["scaler"]
            )
            model = model.to(self.device)
        else:
            model = model.to(self.device)

        for idx, context in enumerate(contexts):
            benchmark_type = benchmark.get_benchmark_dict()[self.benchmark_type]()
            task_bddl_file = benchmark_type.get_task_bddl_file_path(context)
            task_name = os.path.basename(task_bddl_file).split(".")[0]

            task_emb = self.task_embs[task_name].to(self.device).unsqueeze(0)
            init_states = benchmark_type.get_task_init_states(context)

            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=128,
                camera_widths=128,
            )

            model.reset()
            env.seed(self.seed)
            env.reset()
            obs = env.set_init_state(init_states[context_ind[idx]])

            dummy = np.zeros(7)
            dummy[-1] = -1.0
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            video_writer = None
            if self.save_video and self.save_video_dir is not None:
                save_dir = os.path.join(
                    self.save_video_dir,
                    self.benchmark_type,
                    "videos",
                    task_name,
                )
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir, f"episode_{context_ind[idx]}.mp4"
                )
                video_writer = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (1280, 800),
                )

            success_flag = 0
            for step in range(self.max_step_per_episode):
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
                    np.concatenate(
                        [
                            obs["robot0_joint_pos"],
                            obs["robot0_gripper_qpos"],
                        ]
                    )
                ).float().unsqueeze(0).unsqueeze(0).to(self.device)

                if video_writer or self.render_image:
                    img = env.sim.render(
                        camera_name="frontview", width=1280, height=800
                    )[..., ::-1]
                    img = np.flip(img, axis=0)
                    if video_writer:
                        video_writer.write(img)
                    safe_display_image(img, f"frontview_{pid}", self.render_image)

                action = model.predict(
                    {
                        "agentview_image": agentview,
                        "eye_in_hand_image": eye,
                        "lang_emb": task_emb,
                        "robot_states": robot_state,
                    }
                ).cpu().numpy()

                obs, reward, done, _ = env.step(action)

                if reward == 1:
                    success_flag = 1
                    success[context, context_ind[idx]] = 1
                    episode_lengths[context, context_ind[idx]] = step + 1
                    log.info(
                        f"[Task {context:02d} | Episode {context_ind[idx]:02d}] {task_name} - ✓ SUCCESS at step {step + 1}"
                    )
                    break

            if not success_flag:
                episode_lengths[context, context_ind[idx]] = self.max_step_per_episode
                log.info(
                    f"[Task {context:02d} | Episode {context_ind[idx]:02d}] {task_name} - ✗ FAILED (max steps: {self.max_step_per_episode})"
                )

            if video_writer:
                video_writer.release()

            safe_destroy_window(f"frontview_{pid}", self.render_image)
            env.close()

            # Handle both multiprocessing Value and simple dict counter
            if hasattr(counter, 'get_lock'):
                with counter.get_lock():
                    counter.value += 1
            else:
                counter["value"] += 1

    def test_model(self, model, model_config=None, cpu_set=None, epoch=0):
        if "DISPLAY" not in os.environ:
            self.render_image = False

        num_tasks = 50 if self.benchmark_type == "libero_90" else 10
        all_runs = num_tasks * self.rollouts

        success = torch.zeros(num_tasks, self.rollouts)
        episode_lengths = torch.zeros(num_tasks, self.rollouts)

        contexts = np.repeat(np.arange(num_tasks), self.rollouts)
        context_ind = np.tile(np.arange(self.rollouts), num_tasks)

        if self.use_multiprocessing:
            # Use multiprocessing for parallel evaluation
            if model_config is None:
                raise ValueError(
                    "model_config must be provided when use_multiprocessing=True. "
                    "Multiprocessing requires model_config to instantiate models in child processes."
                )
            
            success = success.share_memory_()
            episode_lengths = episode_lengths.share_memory_()
            
            ctx = mp.get_context("spawn")
            counter = ctx.Value("i", 0)

            model_states = model.get_model_state
            shared_states = {
                "model": {k: v.share_memory_() for k, v in model_states[0].items()},
                "scaler": model_states[1],
            }

            processes = []
            splits = np.array_split(np.arange(all_runs), self.n_cores)

            for pid, split in enumerate(splits):
                p = ctx.Process(
                    target=self.eval_model,
                    kwargs=dict(
                        contexts=contexts[split],
                        context_ind=context_ind[split],
                        success=success,
                        episode_lengths=episode_lengths,
                        pid=pid,
                        cpu_set=None,
                        counter=counter,
                        all_runs=all_runs,
                        model=None,
                        model_config=model_config,
                        model_states=shared_states,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            # Run sequentially in main process
            model = model.to(self.device)
            counter = {"value": 0}  # Simple dict instead of multiprocessing Value
            
            # Run all evaluations sequentially
            for idx in range(all_runs):
                self.eval_model(
                    contexts=contexts[idx:idx+1],
                    context_ind=context_ind[idx:idx+1],
                    success=success,
                    episode_lengths=episode_lengths,
                    pid=0,
                    cpu_set=None,
                    counter=counter,
                    all_runs=all_runs,
                    model=model,
                    model_config=None,
                    model_states=None,
                )

        success_rate = torch.mean(success, dim=1)
        avg_success = torch.mean(success_rate).item()
        
        # Get task names for logging
        benchmark_type_obj = benchmark.get_benchmark_dict()[self.benchmark_type]()
        task_names = []
        for task_idx in range(num_tasks):
            task_bddl_file = benchmark_type_obj.get_task_bddl_file_path(task_idx)
            task_name = os.path.basename(task_bddl_file).split(".")[0]
            task_names.append(task_name)
        
        # Print and log success rate for each task
        log.info(f"\n{'='*80}")
        log.info(f"Evaluation Results (Epoch {epoch})")
        log.info(f"{'='*80}")
        log.info(f"{'Task':<50} {'Success Rate':<15} {'Avg Episode Length':<20}")
        log.info(f"{'-'*80}")
        
        wandb_log_dict = {f"epoch{epoch}_average_success": avg_success}
        
        for task_idx in range(num_tasks):
            task_name = task_names[task_idx]
            task_success_rate = success_rate[task_idx].item()
            avg_episode_length = torch.mean(episode_lengths[task_idx]).item()
            
            # Format success rate as percentage
            success_pct = task_success_rate * 100
            
            log.info(f"{task_name:<50} {success_pct:>6.1f}% ({int(success[task_idx].sum().item())}/{self.rollouts})  {avg_episode_length:>6.1f}")
            
            # Log to wandb for each task
            wandb_log_dict[f"epoch{epoch}_task_{task_idx:02d}_{task_name}_success_rate"] = task_success_rate
            wandb_log_dict[f"epoch{epoch}_task_{task_idx:02d}_{task_name}_avg_episode_length"] = avg_episode_length
        
        log.info(f"{'-'*80}")
        log.info(f"{'Overall Average':<50} {avg_success*100:>6.1f}%")
        log.info(f"{'='*80}\n")
        
        # Store results as instance variables for access from callbacks
        self.success_rate = avg_success
        self.success = success
        self.episode_lengths = episode_lengths
        self.task_names = task_names

        wandb.log(wandb_log_dict)
