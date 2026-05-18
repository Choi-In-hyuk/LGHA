"""
Hierarchical MambaVLA 학습 스크립트.

사용법:
  python -m hierarchical_vla.pipeline.train \
      --data_dir /home/choi/libero_object \
      --annotation_path /home/choi/libero_object/libero_object_annotations.pkl \
      --save_dir ./checkpoints/hierarchical \
      --num_epochs 500 \
      --device cuda

전체 흐름:
  1. HierarchicalLiberoDataset 로드
  2. MambaVLA 계층 모드로 생성 (num_action_types=16)
  3. 매 배치: forward(obs, actions, loc_token, action_type)
  4. Flow matching loss로 학습
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "MambaVLA"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataloader import HierarchicalLiberoDataset
from .action_type_mapper import NUM_ACTION_TYPES
from .gt_localizer import NUM_PHASES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def collate_fn(batch):
    obs_list, action_types, loc_tokens, actions_list, masks_list = zip(*batch)

    obs = {
        k: torch.stack([o[k] for o in obs_list], dim=0)
        for k in obs_list[0]
    }
    return (
        obs,
        torch.stack(action_types),
        torch.stack(loc_tokens),
        torch.stack(actions_list),
        torch.stack(masks_list),
    )


def train(
    data_dir: str,
    annotation_path: str,
    save_dir: str,
    num_epochs: int = 500,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    action_seq_len: int = 10,
    latent_dim: int = 256,
    embed_dim: int = 256,
    n_layer: int = 5,
    save_freq: int = 50,
    num_workers: int = 4,
    demos_per_task: int = 50,
    use_phase: bool = True,           # True → phase 모드 (5단계), False → 레거시 (16단계)
    wandb_project: str = None,
    wandb_name: str = "hierarchical_vla",
):
    # ── 1. Dataset ──────────────────────────────────────────────────────────
    dataset = HierarchicalLiberoDataset(
        data_directory=data_dir,
        annotation_path=annotation_path,
        chunk_size=action_seq_len,
        demos_per_task=demos_per_task,
        use_phase=use_phase,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )
    log.info(f"Dataset: {len(dataset)} slices, {len(dataloader)} batches/epoch")

    # ── 2. Model ─────────────────────────────────────────────────────────────
    from MambaVLA.model_factory import create_mambavla_model
    from MambaVLA.utils.scaler import MinMaxScaler

    num_types = NUM_PHASES if use_phase else NUM_ACTION_TYPES
    log.info(f"Model: num_action_types={num_types} ({'phase' if use_phase else 'legacy'})")

    model = create_mambavla_model(
        camera_names=["agentview", "eye_in_hand"],
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        action_seq_len=action_seq_len,
        n_layer=n_layer,
        device=device,
        num_action_types=num_types,
        layer3_channels=256,
    )

    # action scaler (minmax) — fits on all training actions
    all_actions = dataset.get_all_actions()
    scaler = MinMaxScaler(all_actions, scale_data=True, device=device)
    model.set_scaler(scaler)

    if hasattr(model, "configure_optimizers"):
        result = model.configure_optimizers()
        optimizer = result[0] if isinstance(result, (tuple, list)) else result
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ── 3. WandB (optional) ─────────────────────────────────────────────────
    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, name=wandb_name)

    # ── 4. Training loop ─────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:4d}/{num_epochs}", leave=False)
        for obs, action_type, loc_token, actions, mask in pbar:
            # to device
            obs = {k: v.to(device) for k, v in obs.items()}
            action_type = action_type.to(device)
            loc_token   = loc_token.to(device)
            actions     = actions.to(device)

            # scale actions
            actions_scaled = scaler.scale_output(actions)

            optimizer.zero_grad()
            loss = model(
                obs_dict=obs,
                actions=actions_scaled,
                loc_token=loc_token,
                action_type=action_type,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch {epoch:4d}/{num_epochs} | loss={avg_loss:.4f}")

        if use_wandb:
            import wandb
            wandb.log({"train/loss": avg_loss, "epoch": epoch})

        if epoch % save_freq == 0:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch:04d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            model.store_model_scaler(save_dir)
            log.info(f"Saved checkpoint → {ckpt_path}")

    # final save
    model.store_model_weights(save_dir, sv_name="final")
    model.store_model_scaler(save_dir)
    log.info(f"Training complete. Model saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          required=True)
    parser.add_argument("--annotation_path",   required=True)
    parser.add_argument("--save_dir",          default="./checkpoints/hierarchical")
    parser.add_argument("--num_epochs",        type=int,   default=500)
    parser.add_argument("--batch_size",        type=int,   default=256)
    parser.add_argument("--learning_rate",     type=float, default=1e-4)
    parser.add_argument("--device",            default="cuda")
    parser.add_argument("--action_seq_len",    type=int,   default=10)
    parser.add_argument("--save_freq",         type=int,   default=50)
    parser.add_argument("--num_workers",       type=int,   default=4)
    parser.add_argument("--demos_per_task",    type=int,   default=50)
    parser.add_argument("--wandb_project",     default=None)
    parser.add_argument("--wandb_name",        default="hierarchical_vla")
    parser.add_argument("--no_phase",          action="store_true",
                        help="phase 모드 비활성화 (레거시 action_type 사용)")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        annotation_path=args.annotation_path,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        action_seq_len=args.action_seq_len,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
        demos_per_task=args.demos_per_task,
        use_phase=not args.no_phase,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )


if __name__ == "__main__":
    main()
