"""
IDEA2 v2 Stage 1: VQ-VAE 학습.

ResNet cache delta (z_{t+1} - z_t) → discrete codebook index.

사전 조건: precompute_resnet.py 실행 완료.

Usage:
    python -m hierarchical_vla.idea2.train_stage1 \\
        --data_dir /home/choi/LGHA/LIBERO/libero/datasets/libero_10_subset \\
        --cache_dir ./resnet_cache/libero_10_subset \\
        --save_dir ./checkpoints/idea2_v2/stage1 \\
        --K 64 --epochs 200
"""

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import DeltaDataset
from .vqvae import VQVAEModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train(
    data_dir: str,
    cache_dir: str,
    save_dir: str,
    K: int = 64,
    hidden_dim: int = 256,
    beta: float = 0.25,
    epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cuda",
    demos_per_task: int = 50,
    step: int = 10,
):
    os.makedirs(save_dir, exist_ok=True)

    log.info(f"Loading DeltaDataset (step={step})...")
    dataset = DeltaDataset(
        cache_dir=cache_dir,
        data_dir=data_dir,
        demos_per_task=demos_per_task,
        step=step,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    log.info(f"Dataset: {len(dataset)} delta samples")

    input_dim = dataset.deltas.shape[-1]
    model = VQVAEModel(
        input_dim=input_dim, hidden_dim=hidden_dim, K=K, beta=beta
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    log.info(f"VQ-VAE: K={K}, D={hidden_dim}, params={sum(p.numel() for p in model.parameters()):,}")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc="Batches", leave=False):
            delta = batch.to(device)
            _, _, loss = model(delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            log.info(f"Epoch {epoch+1}/{epochs} | loss: {avg_loss:.4f}")

        # codebook 활용도 체크
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                all_deltas = dataset.deltas.to(device)
                indices = model.encode_delta(all_deltas)
                unique = indices.unique().numel()
            log.info(f"  Codebook usage: {unique}/{K} entries active")

    # 저장
    vqvae_path = os.path.join(save_dir, "vqvae.pt")
    torch.save(model.state_dict(), vqvae_path)
    log.info(f"Saved VQ-VAE: {vqvae_path}")

    codebook_path = os.path.join(save_dir, "codebook.pt")
    torch.save(model.vq.codebook.weight.data.cpu(), codebook_path)
    log.info(f"Saved codebook: {codebook_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str, required=True)
    parser.add_argument("--cache_dir",       type=str, required=True)
    parser.add_argument("--save_dir",        type=str, default="./checkpoints/idea2_v2/stage1")
    parser.add_argument("--K",               type=int, default=64)
    parser.add_argument("--epochs",          type=int, default=200)
    parser.add_argument("--batch_size",      type=int, default=512)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--demos_per_task",  type=int, default=50)
    parser.add_argument("--step",            type=int, default=10)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        save_dir=args.save_dir,
        K=args.K,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        demos_per_task=args.demos_per_task,
        step=args.step,
    )
