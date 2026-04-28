"""
VQ-VAE for latent action quantization (IDEA2 Stage 1).

V-JEPA delta (z_{t+1} - z_t) → discrete codebook index.
각 codebook entry = 하나의 "행동 패턴" (scene 변화 유형).

Online codebook update:
  primitive 실행 후 실제 관측된 delta로 해당 entry를 EMA 업데이트.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Standard VQ with straight-through estimator.
    codebook: (K, D) learnable entries.
    """

    def __init__(self, K: int, D: int, beta: float = 0.25):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.codebook = nn.Embedding(K, D)
        nn.init.uniform_(self.codebook.weight, -1.0 / K, 1.0 / K)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: (B, D) encoder output

        Returns:
            z_q_st: (B, D) quantized (straight-through)
            indices: (B,) codebook indices
            vq_loss: scalar
        """
        # (B, K) distances
        dist = (
            z_e.unsqueeze(1) - self.codebook.weight.unsqueeze(0)
        ).pow(2).sum(dim=-1)

        indices = dist.argmin(dim=-1)          # (B,)
        z_q = self.codebook(indices)           # (B, D)

        # VQ loss: commitment loss
        vq_loss = (
            F.mse_loss(z_q.detach(), z_e)
            + self.beta * F.mse_loss(z_q, z_e.detach())
        )

        # Straight-through estimator: gradient flows through z_e
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, indices, vq_loss

    @torch.no_grad()
    def ema_update(self, index: int, target: torch.Tensor, momentum: float = 0.99):
        """
        Online EMA update for a single codebook entry.
        target: (D,) 실제 관측된 delta
        """
        self.codebook.weight[index] = (
            momentum * self.codebook.weight[index]
            + (1 - momentum) * target
        )

    @torch.no_grad()
    def encode(self, z_e: torch.Tensor) -> torch.Tensor:
        """z_e → nearest codebook index. (B, D) → (B,)"""
        dist = (
            z_e.unsqueeze(1) - self.codebook.weight.unsqueeze(0)
        ).pow(2).sum(dim=-1)
        return dist.argmin(dim=-1)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """index → codebook entry. (B,) → (B, D)"""
        return self.codebook(indices)


class VQVAEModel(nn.Module):
    """
    VQ-VAE for latent action quantization.

    Input:  delta = z_{t+chunk} - z_t  (256-dim, V-JEPA space)
    Output: reconstructed delta + codebook index + loss
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        K: int = 256,
        beta: float = 0.25,
    ):
        super().__init__()
        self.K = K

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vq = VectorQuantizer(K, hidden_dim, beta)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, delta: torch.Tensor):
        """
        Args:
            delta: (B, input_dim)

        Returns:
            recon:   (B, input_dim) reconstructed delta
            indices: (B,) codebook indices
            loss:    scalar (recon + vq)
        """
        z_e = self.encoder(delta)
        z_q_st, indices, vq_loss = self.vq(z_e)
        recon = self.decoder(z_q_st)
        recon_loss = F.mse_loss(recon, delta)
        return recon, indices, recon_loss + vq_loss

    @torch.no_grad()
    def encode_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """delta → codebook index (inference용). (B, D) → (B,)"""
        z_e = self.encoder(delta)
        return self.vq.encode(z_e)

    @torch.no_grad()
    def online_update(self, index: int, delta_real: torch.Tensor, momentum: float = 0.99):
        """
        Primitive 실행 후 호출.
        실제 관측된 delta로 codebook entry EMA 업데이트.

        Args:
            index:      int, 이 primitive에 할당된 codebook entry index
            delta_real: (D,) 실제 z_after - z_before
            momentum:   EMA momentum (0.99 권장)
        """
        z_e = self.encoder(delta_real.unsqueeze(0)).squeeze(0)
        self.vq.ema_update(index, z_e, momentum)
