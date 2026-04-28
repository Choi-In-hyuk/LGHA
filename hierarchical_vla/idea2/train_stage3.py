"""
IDEA2 v2 Stage 3: Action Decoder 학습.

codebook[k] + (z_scene_live + context_vec) → robot action sequence (MSE).
z_scene은 scene_encoder로 live 계산 → 학습/inference 완전 일치.

사전 조건: Stage 1 + Stage 2 완료.

Usage:
    python -m hierarchical_vla.idea2.train_stage3 \\
        --data_dir /home/choi/LGHA/LIBERO/libero/datasets/libero_10_subset \\
        --cache_dir ./resnet_cache/libero_10_subset \\
        --vqvae_path ./checkpoints/idea2_v2/stage1/vqvae.pt \\
        --stage2_path ./checkpoints/idea2_v2/stage2/final_model.pth \\
        --save_dir ./checkpoints/idea2_v2/stage3 \\
        --epochs 1000
"""

import argparse
import logging
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from MambaVLA.utils.scaler import MinMaxScaler

from .dataset import LatentActionDataset
from .vqvae import VQVAEModel
from .model_factory import create_idea2_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train(
    data_dir: str,
    cache_dir: str,
    vqvae_path: str,
    stage2_path: str,
    save_dir: str,
    K: int = 64,
    scene_emb_dim: int = 256,
    embed_dim: int = 256,
    n_layer: int = 5,
    d_intermediate: int = 256,
    action_seq_len: int = 10,
    max_len_data: int = 400,
    demos_per_task: int = 50,
    batch_size: int = 64,
    epochs: int = 1000,
    lr: float = 1e-4,
    save_freq: int = 100,
    device: str = "cuda",
    num_workers: int = 0,
):
    os.makedirs(save_dir, exist_ok=True)

    # ── VQ-VAE 로드 ──────────────────────────────────────────────────────
    vqvae = VQVAEModel(input_dim=scene_emb_dim, hidden_dim=scene_emb_dim, K=K)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location="cpu"))
    vqvae.eval()

    # ── Dataset ──────────────────────────────────────────────────────────
    log.info("Building LatentActionDataset...")
    dataset = LatentActionDataset(
        cache_dir=cache_dir,
        vqvae=vqvae,
        data_directory=data_dir,
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
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )

    # ── Scaler ───────────────────────────────────────────────────────────
    all_actions = dataset.get_all_actions()
    scaler = MinMaxScaler(all_actions, scale_data=True, device=device)

    # ── Model ─────────────────────────────────────────────────────────────
    scene_encoder_path = os.path.join(cache_dir, "scene_encoder.pt")
    model = create_idea2_model(
        K=K,
        scene_emb_dim=scene_emb_dim,
        embed_dim=embed_dim,
        n_layer=n_layer,
        d_intermediate=d_intermediate,
        action_seq_len=action_seq_len,
        device=device,
        vqvae_path=vqvae_path,
        scene_encoder_path=scene_encoder_path,
    )
    model.set_scaler(scaler)

    # Stage 2 가중치 로드
    if stage2_path and os.path.exists(stage2_path):
        ckpt = torch.load(stage2_path, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        log.info(f"Loaded Stage 2 weights: {stage2_path}")

    # Stage 3: action_decoder만 학습, 나머지 frozen
    for p in model.parameters():
        p.requires_grad = False
    for p in model.action_decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.action_decoder.parameters(), lr=lr, weight_decay=1e-4
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable params (decoder only): {trainable:,}")

    # ── Training ─────────────────────────────────────────────────────────
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.action_decoder.train()

        total_loss = 0.0
        for batch in tqdm(dataloader, desc="Batches", leave=False):
            obs_raw, actions, mask, z_scene, codebook_idx, context_vec = batch

            agentview   = obs_raw["agentview_image"].to(device).float()
            eye_in_hand = obs_raw["eye_in_hand_image"].to(device).float()
            lang_emb    = obs_raw["lang_emb"].to(device).float()
            actions     = actions.to(device).float()
            target_idx  = codebook_idx.to(device)
            context_vec = context_vec.to(device).float()

            actions = scaler.scale_output(actions)

            if lang_emb.dim() == 2:
                lang_emb = lang_emb.unsqueeze(1)

            obs_dict = {
                "agentview_image":   agentview,
                "eye_in_hand_image": eye_in_hand,
                "lang_emb":          lang_emb,
            }

            optimizer.zero_grad()
            # z_scene: 캐시에서 직접 전달 → V-JEPA live forward 생략 (빠른 학습)
            # scene_cache는 동일한 scene_encoder.pt로 만들어졌으므로 일관성 유지
            loss = model.forward_stage3(
                obs_dict, actions, target_idx,
                context_vec=context_vec,
                z_scene=z_scene.to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.action_decoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        log.info(f"Epoch {epoch+1}/{epochs} | MSE loss: {avg_loss:.4f}")

        if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1:05d}.pth")
            trainable_state = {
                k: v for k, v in model.state_dict().items()
                if not k.startswith("scene_encoder.")
            }
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": trainable_state, "loss": avg_loss},
                ckpt_path,
            )
            scaler_path = os.path.join(save_dir, "model_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            log.info(f"Saved: {ckpt_path}")

    final_path = os.path.join(save_dir, "final_model.pth")
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("scene_encoder.")
    }
    torch.save(trainable_state, final_path)
    log.info(f"Stage 3 complete. Final: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str, required=True)
    parser.add_argument("--cache_dir",       type=str, required=True)
    parser.add_argument("--vqvae_path",      type=str, required=True)
    parser.add_argument("--stage2_path",     type=str, required=True)
    parser.add_argument("--save_dir",        type=str, default="./checkpoints/idea2_v2/stage3")
    parser.add_argument("--K",               type=int, default=64)
    parser.add_argument("--epochs",          type=int, default=1000)
    parser.add_argument("--batch_size",      type=int, default=64)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--save_freq",       type=int, default=100)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--num_workers",     type=int, default=0)
    parser.add_argument("--demos_per_task",  type=int, default=50)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        vqvae_path=args.vqvae_path,
        stage2_path=args.stage2_path,
        save_dir=args.save_dir,
        K=args.K,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_freq=args.save_freq,
        device=args.device,
        num_workers=args.num_workers,
        demos_per_task=args.demos_per_task,
    )
