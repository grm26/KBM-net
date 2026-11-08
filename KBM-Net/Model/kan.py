from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class KANModalEncoder(nn.Module):
    """Encode each modality independently with a compact KAN encoder."""

    def __init__(self, feature_list, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.feature_list = feature_list
        self.modal_encoders = nn.ModuleList(
            [
                KANEncode(
                    in_channels=len(feature_indices),
                    hidden_channels=hidden_dim,
                    out_channels=embed_dim,
                )
                for feature_indices in feature_list
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        modality_embeddings = []
        for encoder, feature_indices in zip(
            self.modal_encoders,
            self.feature_list,
        ):
            modality_features = features[:, feature_indices]
            _, embedding = encoder(modality_features)
            modality_embeddings.append(embedding)
        return torch.stack(modality_embeddings, dim=1)


class KANEncode(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.kan1 = KANLinear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(0.1)
        self.kan2 = KANLinear(hidden_channels, out_channels)

    def forward(self, features):
        hidden = self.kan1(features)
        hidden = F.silu(self.norm(hidden))
        hidden = self.dropout(hidden)
        embedding = self.kan2(hidden)
        return hidden, embedding


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 1,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=F.selu,
        grid_eps: float = 0.002,
        grid_range=(-5.0, 5.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(
                -spline_order,
                grid_size + spline_order + 1,
                dtype=torch.float32,
            )
            * step
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                grid_size + spline_order,
            )
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.empty(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(
            self.base_weight,
            gain=math.sqrt(1.0) * self.scale_base,
        )

        with torch.no_grad():
            noise = (
                (
                    torch.rand(
                        self.grid_size + 1,
                        self.in_features,
                        self.out_features,
                    )
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.copy_(
                (
                    self.scale_spline
                    if not self.enable_standalone_scale_spline
                    else 1.0
                )
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.xavier_normal_(
                    self.spline_scaler,
                    gain=math.sqrt(1.0) * self.scale_spline,
                )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected [batch, {self.in_features}], got {tuple(x.shape)}"
            )

        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(order + 1)])
                / (grid[:, order:-1] - grid[:, : -(order + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, order + 1 :] - x)
                / (grid[:, order + 1 :] - grid[:, 1:-order])
                * bases[:, :, 1:]
            )

        expected_shape = (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        if bases.size() != expected_shape:
            raise RuntimeError(
                f"Unexpected spline basis shape {tuple(bases.shape)}, "
                f"expected {expected_shape}"
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError("Invalid spline interpolation input shape.")
        if y.size() != (x.size(0), self.in_features, self.out_features):
            raise ValueError("Invalid spline interpolation target shape.")

        basis = self.b_splines(x).transpose(0, 1)
        targets = y.transpose(0, 1)
        solution = torch.linalg.lstsq(basis, targets).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline:
            scale = self.spline_scaler.unsqueeze(-1)
        else:
            scale = 1.0
        return self.spline_weight * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected [batch, {self.in_features}], got {tuple(x.shape)}"
            )

        base_output = F.linear(
            self.base_activation(x),
            self.base_weight,
        )
        spline_output = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )
        return base_output + spline_output
