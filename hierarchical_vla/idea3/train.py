"""
IDEA3 Fine-tuning: MambaVLAWithComp.

기존 MambaVLA에 comp_proj (512→256) 레이어를 추가해서 fine-tune.

학습 전략:
  - comp_delta = zero (50%): pretrained 동작 보존
  - comp_delta = random delta (50%): 환경 변화 반영 능력 학습
  - Mamba backbone + comp_proj 학습, img_encoder frozen

Usage:
    python -m hierarchical_vla.idea3.train \\
        --checkpoint /home/choi/LGHA/hierarchical_vla/checkpoints/libero_10_subset/final_model.pth \\
        --data_dir /home/choi/LGHA/LIBERO/libero/datasets/libero_10_subset \\
        --save_dir ./checkpoints/idea3 \\
        --epochs 200
"""

import argparse
import logging
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from MambaVLA.model_factory import create_mambavla_model
from MambaVLA.utils.scaler import MinMaxScaler

from .vla_with_comp import MambaVLAWithComp
from .dataset import CompDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CAM_NAMES   = ["agentview", "eye_in_hand"]
LANG_EMB_DIR = os.path.join(os.path.dirname(__file__), "..", "language_embeddings")


def load_base_model(checkpoint_path: str, device: str) -> MambaVLAWithComp:
    """
    create_mambavla_model로 MambaVLA 생성 → 클래스를 MambaVLAWithComp로 교체.
    comp_proj, comp_norm, _comp_delta 속성을 추가하고 체크포인트 로드.
    """
    from MambaVLA.model_factory import create_mambavla_model

    # 기존 아키텍처 그대로 생성
    base = create_mambavla_model(
        camera_names=CAM_NAMES,
        latent_dim=256,
        action_dim=7,
        lang_emb_dim=512,
        embed_dim=256,
        obs_tok_len=len(CAM_NAMES),
        action_seq_len=10,
        perception_seq_len=1,
        device=device,
        n_layer=5,
        d_intermediate=256,
        state_dim=45,
        use_language_encoder=True,
        freeze_language_encoder=True,
    )

    # 클래스 교체: MambaVLA → MambaVLAWithComp
    base.__class__ = MambaVLAWithComp

    # comp 속성 추가
    comp_input_dim = MambaVLAWithComp.NUM_CAMS * MambaVLAWithComp.LATENT_DIM  # 512
    base.comp_norm = nn.LayerNorm(comp_input_dim).to(device)
    base.comp_proj = nn.Linear(comp_input_dim, MambaVLAWithComp.LATENT_DIM).to(device)
    nn.init.zeros_(base.comp_proj.weight)
    nn.init.zeros_(base.comp_proj.bias)
    base._comp_delta = None

    # pretrained 가중치 로드 (comp_* 키는 없으므로 strict=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, _ = base.load_state_dict(state, strict=False)
    comp_missing = [k for k in missing if "comp" not in k]
    if comp_missing:
        log.warning(f"Missing non-comp keys: {comp_missing}")
    log.info("Loaded pretrained weights into MambaVLAWithComp")

    # scaler 로드
    scaler_path = os.path.join(os.path.dirname(checkpoint_path), "model_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            base.set_scaler(pickle.load(f))
        log.info(f"Loaded scaler: {scaler_path}")
    else:
        log.warning("Scaler not found.")

    return base


def train(
    checkpoint_path: str,
    data_dir: str,
    save_dir: str,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    demos_per_task: int = 50,
    action_seq_len: int = 10,
    comp_prob: float = 0.5,
    ref_horizon: int = 50,
    save_freq: int = 50,
    num_workers: int = 0,
):
    os.makedirs(save_dir, exist_ok=True)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    log.info(f"Loading MambaVLAWithComp from {checkpoint_path}")
    model = load_base_model(checkpoint_path, device)

    # Scaler 로드
    scaler_path = os.path.join(os.path.dirname(checkpoint_path), "model_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            model.set_scaler(pickle.load(f))
        log.info("Loaded scaler")
    else:
        log.warning("Scaler not found. Will compute from data.")

    # ── 학습 파라미터 설정 ────────────────────────────────────────────────────
    # img_encoder frozen, 나머지 fine-tune
    for p in model.img_encoder.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable: {sum(p.numel() for p in trainable_params):,} params")
    log.info(f"  (comp_proj: {sum(p.numel() for p in model.comp_proj.parameters()):,})")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # ── 언어 임베딩 로드 ──────────────────────────────────────────────────────
    emb_path = os.path.join(LANG_EMB_DIR, "libero_10_subset.pkl")
    with open(emb_path, "rb") as f:
        raw = pickle.load(f)
    lang_embs = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))
                 for k, v in raw.items()}

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    log.info("Building CompDataset (precomputing obs_embeds)...")
    dataset = CompDataset(
        data_dir=data_dir,
        lang_embs=lang_embs,
        img_encoder=model.img_encoder,
        device=device,
        action_seq_len=action_seq_len,
        demos_per_task=demos_per_task,
        comp_prob=comp_prob,
        ref_horizon=ref_horizon,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )

    # Scaler 없으면 데이터에서 계산
    if model.scaler is None:
        all_actions = torch.cat([demo["actions"] for demo in dataset.samples], dim=0)
        scaler = MinMaxScaler(all_actions, scale_data=True, device=device)
        model.set_scaler(scaler)
        log.info("Computed scaler from data")

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        model.img_encoder.eval()  # img_encoder frozen

        total_loss = 0.0

        for agentview, eye_in_hand, lang_emb, comp_delta, actions in tqdm(dataloader, leave=False):
            agentview   = agentview.squeeze(1).to(device)    # (B, C, H, W) → unsqueeze later
            eye_in_hand = eye_in_hand.squeeze(1).to(device)
            lang_emb    = lang_emb.to(device)
            comp_delta  = comp_delta.to(device)              # (B, num_cams, latent_dim)
            actions     = actions.to(device)

            # perception_seq_len=1 → (B, 1, C, H, W)
            agentview   = agentview.unsqueeze(1)
            eye_in_hand = eye_in_hand.unsqueeze(1)

            if lang_emb.dim() == 2:
                lang_emb = lang_emb.unsqueeze(1)  # (B, 1, 512)

            actions_scaled = model.scaler.scale_output(actions)

            obs_dict = {
                "agentview_image":   agentview,
                "eye_in_hand_image": eye_in_hand,
                "lang_emb":          lang_emb,
                "comp_delta":        comp_delta,  # ← 핵심: MambaVLAWithComp가 읽음
            }

            optimizer.zero_grad()
            loss = model(obs_dict, actions_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        log.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % save_freq == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1:05d}.pth")
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "loss": avg_loss}, ckpt_path)
            with open(os.path.join(save_dir, "model_scaler.pkl"), "wb") as f:
                pickle.dump(model.scaler, f)
            log.info(f"Saved: {ckpt_path}")

    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    with open(os.path.join(save_dir, "model_scaler.pkl"), "wb") as f:
        pickle.dump(model.scaler, f)
    log.info(f"Done. Final: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--data_dir",      type=str, required=True)
    parser.add_argument("--save_dir",      type=str, default="./checkpoints/idea3")
    parser.add_argument("--epochs",        type=int, default=200)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--device",        type=str, default="cuda")
    parser.add_argument("--demos_per_task",type=int, default=50)
    parser.add_argument("--comp_prob",     type=float, default=0.5)
    parser.add_argument("--ref_horizon",   type=int, default=50)
    parser.add_argument("--save_freq",     type=int, default=50)
    parser.add_argument("--num_workers",   type=int, default=0)
    args = parser.parse_args()

    train(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        demos_per_task=args.demos_per_task,
        comp_prob=args.comp_prob,
        ref_horizon=args.ref_horizon,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
    )
