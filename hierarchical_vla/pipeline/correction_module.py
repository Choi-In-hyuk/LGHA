"""
Correction Module: 파지(grasp) 실패 시 loc_token 보정.

Stage 1 — 데이터 수집:
  evaluate.py 실행 중 grasp 실패 순간의 상태를 JSONL로 기록.
  각 레코드: eef_pos, obj_pos, loc_token_used, gt_loc_token, phase, success

Stage 2 — 학습:
  python -m hierarchical_vla.pipeline.correction_module \
      --data ./logs/grasp_failures.jsonl \
      --out  ./checkpoints/correction

추론 (evaluate.py --correction_module PATH):
  CorrectionModule.correct(eef_pos, obj_pos, wrong_loc_token) → corrected_loc_token
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

N_TOKENS  = 1024
TOKEN_EMB = 16   # loc_token embedding dim


class CorrectionModule(nn.Module):
    """
    (error_vec, wrong_loc_token) → corrected_loc_token 예측 MLP.

    error_vec = eef_pos - obj_pos  (grasp 실패 시 3D 오차, 단위: m)
    wrong_loc_token                (실패 시 사용된 loc_token, 0-1023)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.token_emb = nn.Embedding(N_TOKENS, TOKEN_EMB)
        self.mlp = nn.Sequential(
            nn.Linear(3 + TOKEN_EMB, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_TOKENS),
        )

    def forward(self, error_vec: torch.Tensor, loc_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            error_vec : [B, 3]  float32  (eef_pos - obj_pos)
            loc_token : [B]     int64    (0-1023)
        Returns:
            logits    : [B, N_TOKENS]
        """
        emb = self.token_emb(loc_token)
        x   = torch.cat([error_vec, emb], dim=-1)
        return self.mlp(x)

    @torch.no_grad()
    def correct(
        self,
        eef_pos:   np.ndarray,
        obj_pos:   np.ndarray,
        loc_token: int,
        device:    str = "cpu",
    ) -> int:
        """grasp 실패 후 보정된 loc_token 반환."""
        error_vec = torch.tensor(eef_pos - obj_pos, dtype=torch.float32).unsqueeze(0).to(device)
        tok       = torch.tensor([loc_token], dtype=torch.long).to(device)
        logits    = self.forward(error_vec, tok)
        return int(logits.argmax(dim=-1).item())

    @classmethod
    def load(cls, path: str, device: str = "cpu", hidden_dim: int = 128) -> "CorrectionModule":
        m = cls(hidden_dim=hidden_dim)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device).eval()
        return m


# ── Training ──────────────────────────────────────────────────────────────────

def train_correction_module(
    data_path:  str,
    out_dir:    str   = "./checkpoints/correction",
    hidden_dim: int   = 128,
    epochs:     int   = 100,
    batch_size: int   = 64,
    lr:         float = 1e-3,
    device:     str   = "cuda",
) -> CorrectionModule:
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records in {data_path}")

    log.info(f"Loaded {len(records)} grasp failure records from {data_path}")

    error_vecs = torch.tensor(
        [np.array(r["eef_pos"]) - np.array(r["obj_pos"]) for r in records],
        dtype=torch.float32,
    )
    loc_tokens = torch.tensor([r["loc_token_used"] for r in records], dtype=torch.long)
    gt_tokens  = torch.tensor([r["gt_loc_token"]   for r in records], dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(error_vecs, loc_tokens, gt_tokens)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model   = CorrectionModule(hidden_dim=hidden_dim).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for ev, lt, gt in loader:
            ev, lt, gt = ev.to(device), lt.to(device), gt.to(device)
            logits = model(ev, lt)
            loss   = loss_fn(logits, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            log.info(f"Epoch {epoch+1:4d}/{epochs} | loss={total_loss/len(loader):.4f}")

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "correction_module.pth")
    torch.save(model.state_dict(), save_path)
    log.info(f"Saved → {save_path}")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train CorrectionModule from grasp failure data")
    parser.add_argument("--data",       required=True, help="JSONL 실패 데이터 경로")
    parser.add_argument("--out",        default="./checkpoints/correction")
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_correction_module(
        data_path=args.data,
        out_dir=args.out,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
