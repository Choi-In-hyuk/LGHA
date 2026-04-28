"""
MambaVLAWorld 학습 스크립트.

두 가지 loss를 동시에 학습:
  1. world_loss  : predictor(z_t, lang) vs z_T (goal scene latent)
  2. policy_loss : flow matching with GT goal conditioning

Usage:
    python -m libero_bench.train_world \
        data_directory=/home/choi/LGHA/LIBERO/libero/datasets/libero_10_subset \
        save_dir=./checkpoints/libero_10_subset_world \
        num_epochs=2000 \
        max_len_data=400 \
        num_workers=0 \
        wandb.enabled=false
"""

import logging
import os
import pickle
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .dataloader import sim_framework_path
from MambaVLA.utils.scaler import MinMaxScaler
from hierarchical_vla.world_model.dataset import WorldModelDataset
from hierarchical_vla.world_model.model_factory_world import create_mambavla_world_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    data_directory: str,
    batch_size: int = 64,
    num_epochs: int = 2000,
    learning_rate: float = 1e-4,
    device: str = None,
    latent_dim: int = 256,
    embed_dim: int = 256,
    obs_tok_len: int = 2,
    action_seq_len: int = 10,
    save_dir: str = "./checkpoints",
    save_freq: int = 10,
    max_len_data: int = 400,
    num_workers: int = 0,
    demos_per_task: int = 50,
    sampling_steps: int = 10,
    world_loss_weight: float = 1.0,
    # V-JEPA
    vjepa_model_name: str = "facebook/vjepa2-vitg-fpc64-256",
    vjepa_n_finetune_layers: int = 4,
    vjepa_n_frames: int = 4,
    scene_emb_dim: int = 256,
    # wandb
    wandb_project: str = "MambaVLA",
    wandb_entity: str = "your_wandb_entity",
    wandb_name: str = "mambavla_world",
    wandb_enabled: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info(f"Loading dataset from {data_directory}")
    dataset = WorldModelDataset(
        data_directory=data_directory,
        device="cpu",
        obs_dim=32,
        action_dim=7,
        state_dim=45,
        max_len_data=max_len_data,
        chunck_size=action_seq_len,
        start_idx=0,
        demos_per_task=demos_per_task,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    logger.info(f"Dataset size: {len(dataset)} samples")

    # ── Scaler ────────────────────────────────────────────────────────────
    all_actions = dataset.get_all_actions() if hasattr(dataset, "get_all_actions") \
                  else torch.zeros(100, 7)
    scaler = MinMaxScaler(all_actions, scale_data=True, device=device)

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Building MambaVLAWorld model...")
    model = create_mambavla_world_model(
        dataloader=dataset,
        camera_names=dataset.camera_names,
        latent_dim=latent_dim,
        action_dim=7,
        lang_emb_dim=512,
        scene_emb_dim=scene_emb_dim,
        embed_dim=embed_dim,
        obs_tok_len=obs_tok_len,
        action_seq_len=action_seq_len,
        device=device,
        sampling_steps=sampling_steps,
        learning_rate=learning_rate,
        world_loss_weight=world_loss_weight,
        vjepa_model_name=vjepa_model_name,
        vjepa_n_finetune_layers=vjepa_n_finetune_layers,
        vjepa_n_frames=vjepa_n_frames,
    )
    model.set_scaler(scaler)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = model.configure_optimizer()

    # ── wandb ─────────────────────────────────────────────────────────────
    if wandb_enabled:
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        total_loss_sum = world_loss_sum = policy_loss_sum = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc="Batches", leave=False):
            obs_dict, actions, mask = batch

            agentview   = obs_dict["agentview_image"].to(device).float()
            eye_in_hand = obs_dict["eye_in_hand_image"].to(device).float()
            lang_emb    = obs_dict["lang_emb"].to(device).float()
            goal_frame  = obs_dict["goal_frame"].to(device).float()   # (B, C, H, W)
            actions     = actions.to(device).float()

            actions = scaler.scale_input(actions)

            if lang_emb.dim() == 2:
                lang_emb = lang_emb.unsqueeze(1)

            model_obs = {
                "agentview_image":  agentview,
                "eye_in_hand_image": eye_in_hand,
                "lang_emb":         lang_emb,
            }

            optimizer.zero_grad()
            total_loss, world_loss, policy_loss = model(
                model_obs, actions=actions, goal_frame=goal_frame
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_sum  += total_loss.item()
            world_loss_sum  += world_loss.item()
            policy_loss_sum += policy_loss.item()
            n_batches += 1

        n = max(n_batches, 1)
        avg_total  = total_loss_sum  / n
        avg_world  = world_loss_sum  / n
        avg_policy = policy_loss_sum / n

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"total: {avg_total:.4f}  world: {avg_world:.4f}  policy: {avg_policy:.4f}"
        )

        if wandb_enabled:
            wandb.log({
                "train_loss":        avg_total,
                "world_loss":        avg_world,
                "policy_loss":       avg_policy,
                "epoch":             epoch + 1,
            })

        if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1:05d}.pth")
            torch.save({
                "epoch":               epoch + 1,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                avg_total,
            }, ckpt_path)
            scaler_path = os.path.join(save_dir, "model_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Final model: {final_path}")

    if wandb_enabled:
        wandb.finish()


_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf")

@hydra.main(version_base=None, config_path=_config_path, config_name="config")
def main(cfg: DictConfig) -> None:
    train(
        data_directory=cfg.data_directory,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        device=cfg.get("device", None),
        latent_dim=cfg.latent_dim,
        embed_dim=cfg.embed_dim,
        obs_tok_len=cfg.obs_tok_len,
        action_seq_len=cfg.action_seq_len,
        save_dir=cfg.save_dir,
        save_freq=cfg.save_freq,
        max_len_data=cfg.max_len_data,
        num_workers=cfg.num_workers,
        demos_per_task=cfg.demos_per_task,
        sampling_steps=cfg.sampling_steps,
        world_loss_weight=cfg.get("world_loss_weight", 1.0),
        vjepa_model_name=cfg.get("vjepa_model_name", "facebook/vjepa2-vitg-fpc64-256"),
        vjepa_n_finetune_layers=cfg.get("vjepa_n_finetune_layers", 4),
        vjepa_n_frames=cfg.get("vjepa_n_frames", 4),
        scene_emb_dim=cfg.get("scene_emb_dim", 256),
        wandb_project=cfg.wandb.project,
        wandb_entity=cfg.wandb.entity,
        wandb_name=cfg.get("wandb", {}).get("name", "mambavla_world"),
        wandb_enabled=cfg.wandb.enabled,
    )


if __name__ == "__main__":
    main()
