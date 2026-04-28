"""
MambaVLAScene 학습 스크립트.

캐시 모드 (권장, 빠름):
    1. 먼저 V-JEPA 임베딩 사전 계산:
       python -m models.precompute_vjepa \
           --data_dir .../libero_10_subset \
           --cache_dir ./vjepa_cache/libero_10_subset

    2. 캐시로 학습:
       python -m libero_bench.train_scene \
           data_directory=.../libero_10_subset \
           save_dir=./checkpoints/libero_10_subset_scene \
           vjepa_cache_dir=./vjepa_cache/libero_10_subset \
           num_epochs=2000 max_len_data=400 num_workers=0 wandb.enabled=false

라이브 모드 (느림, 캐시 없을 때):
    python -m libero_bench.train_scene \
        data_directory=.../libero_10_subset \
        save_dir=./checkpoints/libero_10_subset_scene \
        num_epochs=2000 max_len_data=400 num_workers=0 batch_size=16 wandb.enabled=false
"""

import logging
import os
import pickle
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import hydra
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataloader import LiberoDataset, sim_framework_path
from MambaVLA.utils.scaler import MinMaxScaler
from hierarchical_vla.models.model_factory_scene import create_mambavla_scene_model

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
    n_layer: int = 5,
    d_intermediate: int = 256,
    obs_tok_len: int = 2,
    action_seq_len: int = 10,
    save_dir: str = "./checkpoints",
    save_freq: int = 10,
    max_len_data: int = 400,
    num_workers: int = 0,
    demos_per_task: int = 50,
    sampling_steps: int = 4,
    # V-JEPA
    vjepa_model_name: str = "facebook/vjepa2-vitg-fpc64-256",
    vjepa_n_finetune_layers: int = 4,
    vjepa_n_frames: int = 1,
    scene_emb_dim: int = 256,
    vjepa_cache_dir: str = None,   # 캐시 디렉토리 (있으면 캐시 모드)
    # wandb
    wandb_project: str = "MambaVLA",
    wandb_entity: str = "your_wandb_entity",
    wandb_name: str = "mambavla_scene",
    wandb_enabled: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    use_cache = vjepa_cache_dir is not None
    logger.info(f"Loading dataset (cache_mode={use_cache})")

    if use_cache:
        from hierarchical_vla.models.cached_scene_dataset import CachedSceneDataset
        dataset = CachedSceneDataset(
            cache_dir=vjepa_cache_dir,
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
    else:
        dataset = LiberoDataset(
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
    all_actions = dataset.get_all_actions()
    scaler = MinMaxScaler(all_actions, scale_data=True, device=device)

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Building MambaVLAScene model...")
    model = create_mambavla_scene_model(
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
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        sampling_steps=sampling_steps,
        learning_rate=learning_rate,
        vjepa_model_name=vjepa_model_name,
        vjepa_n_finetune_layers=0 if use_cache else vjepa_n_finetune_layers,
        vjepa_n_frames=vjepa_n_frames,
    )
    model.set_scaler(scaler)
    model.train()

    # 캐시 모드에서 V-JEPA는 완전 frozen → GPU 메모리 절약
    if use_cache:
        model.scene_encoder.eval()
        for p in model.scene_encoder.parameters():
            p.requires_grad = False
        logger.info("Cache mode: V-JEPA fully frozen (scene encoder bypassed in forward)")

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
        if use_cache:
            model.scene_encoder.eval()

        epoch_loss = 0.0
        n_batches  = 0

        for batch in tqdm(dataloader, desc="Batches", leave=False):
            obs_raw, actions, mask = batch

            agentview   = obs_raw["agentview_image"].to(device).float()
            eye_in_hand = obs_raw["eye_in_hand_image"].to(device).float()
            lang_emb    = obs_raw["lang_emb"].to(device).float()
            actions     = actions.to(device).float()

            actions = scaler.scale_output(actions)

            if lang_emb.dim() == 2:
                lang_emb = lang_emb.unsqueeze(1)

            obs_dict = {
                "agentview_image":   agentview,
                "eye_in_hand_image": eye_in_hand,
                "lang_emb":          lang_emb,
            }

            # 캐시 모드: z_scene을 캐시에서 주입
            if use_cache:
                z_scene = obs_raw["z_scene_cache"].to(device).float()  # (B, D)
                obs_dict["z_scene_cache"] = z_scene.unsqueeze(1)       # (B, 1, D)

            optimizer.zero_grad()
            loss = model(obs_dict, actions=actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch+1}/{num_epochs} | loss: {avg_loss:.4f}")

        if wandb_enabled:
            wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

        if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1:05d}.pth")
            # V-JEPA는 frozen이므로 저장 제외 (HuggingFace에서 재로드)
            trainable_state = {k: v for k, v in model.state_dict().items()
                               if not k.startswith("scene_encoder.model.")}
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     trainable_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                 avg_loss,
            }, ckpt_path)
            scaler_path = os.path.join(save_dir, "model_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(save_dir, "final_model.pth")
    trainable_state = {k: v for k, v in model.state_dict().items()
                       if not k.startswith("scene_encoder.model.")}
    torch.save(trainable_state, final_path)
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
        n_layer=cfg.n_layer,
        d_intermediate=cfg.d_intermediate,
        obs_tok_len=cfg.obs_tok_len,
        action_seq_len=cfg.action_seq_len,
        save_dir=cfg.save_dir,
        save_freq=cfg.save_freq,
        max_len_data=cfg.max_len_data,
        num_workers=cfg.num_workers,
        demos_per_task=cfg.demos_per_task,
        sampling_steps=cfg.sampling_steps,
        vjepa_model_name=cfg.get("vjepa_model_name", "facebook/vjepa2-vitg-fpc64-256"),
        vjepa_n_finetune_layers=cfg.get("vjepa_n_finetune_layers", 4),
        vjepa_n_frames=cfg.get("vjepa_n_frames", 1),
        scene_emb_dim=cfg.get("scene_emb_dim", 256),
        vjepa_cache_dir=cfg.get("vjepa_cache_dir", None),
        wandb_project=cfg.wandb.project,
        wandb_entity=cfg.wandb.entity,
        wandb_name=cfg.get("wandb", {}).get("name", "mambavla_scene"),
        wandb_enabled=cfg.wandb.enabled,
    )


if __name__ == "__main__":
    main()
