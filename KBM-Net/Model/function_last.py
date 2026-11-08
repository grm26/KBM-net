from __future__ import annotations

import math

import torch
from einops import rearrange, repeat
from torch import nn

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class Cross_Modal_Mamba(nn.Module):
    """Cross-modal selective scan with mutual C-parameter exchange.

    The X branch uses C_y generated from Y, and the Y branch uses C_x
    generated from X. The internal width is expanded by ``ssm_ratio`` and
    projected back to ``d_model`` before the residual connection.
    """

    def __init__(
        self,
        d_model: int = 96,
        d_state: int = 4,
        ssm_ratio: float = 1.0,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = (
            math.ceil(d_model / 6) if d_state == "auto" else int(d_state)
        )
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = (
            math.ceil(d_model / 16)
            if dt_rank == "auto"
            else int(dt_rank)
        )

        self.proj_in_x = nn.Linear(d_model, self.d_inner)
        self.proj_in_y = nn.Linear(d_model, self.d_inner)

        self.conv_x = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
        )
        self.conv_y = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
        )
        self.activation_x = nn.SiLU()
        self.activation_y = nn.SiLU()

        parameter_dim = self.dt_rank + 2 * self.d_state
        self.x_proj = nn.Linear(
            self.d_inner,
            parameter_dim,
            bias=False,
        )
        self.y_proj = nn.Linear(
            self.d_inner,
            parameter_dim,
            bias=False,
        )

        self.dt_proj_x = self.dt_init(
            self.dt_rank,
            self.d_inner,
            dt_scale,
            dt_init,
            dt_min,
            dt_max,
            dt_init_floor,
        )
        self.dt_proj_y = self.dt_init(
            self.dt_rank,
            self.d_inner,
            dt_scale,
            dt_init,
            dt_min,
            dt_max,
            dt_init_floor,
        )

        self.A_log_x = self.A_log_init(self.d_state, self.d_inner)
        self.A_log_y = self.A_log_init(self.d_state, self.d_inner)
        self.D_x = self.D_init(self.d_inner)
        self.D_y = self.D_init(self.d_inner)

        self.out_proj_x = nn.Linear(self.d_inner, d_model)
        self.out_proj_y = nn.Linear(self.d_inner, d_model)

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale

        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(
                dt_proj.weight,
                -dt_init_std,
                dt_init_std,
            )
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inverse_softplus_dt)
        return dt_proj

    @staticmethod
    def A_log_init(
        d_state: int,
        d_inner: int,
    ) -> nn.Parameter:
        values = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        parameter = nn.Parameter(torch.log(values))
        parameter._no_weight_decay = True
        return parameter

    @staticmethod
    def D_init(d_inner: int) -> nn.Parameter:
        parameter = nn.Parameter(torch.ones(d_inner))
        parameter._no_weight_decay = True
        return parameter

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("Cross_Modal_Mamba expects [batch, tokens, channels].")
        if x.shape != y.shape:
            raise ValueError(
                f"Cross-modal inputs must have the same shape, got "
                f"{tuple(x.shape)} and {tuple(y.shape)}."
            )
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected channel dimension {self.d_model}, got {x.size(-1)}."
            )

        residual_x = x
        residual_y = y
        x = self.activation_x(
            self.conv_x(
                rearrange(self.proj_in_x(x), "b l d -> b d l")
            )
        )
        y = self.activation_y(
            self.conv_y(
                rearrange(self.proj_in_y(y), "b l d -> b d l")
            )
        )

        batch_size, _, sequence_length = x.shape
        x_parameters = self.x_proj(
            rearrange(x, "b d l -> (b l) d")
        )
        y_parameters = self.y_proj(
            rearrange(y, "b d l -> (b l) d")
        )

        dt_x, B_x, C_x = torch.split(
            x_parameters,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt_y, B_y, C_y = torch.split(
            y_parameters,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        dt_x = self.dt_proj_x.weight @ dt_x.t()
        dt_y = self.dt_proj_y.weight @ dt_y.t()
        dt_x = rearrange(
            dt_x,
            "d (b l) -> b d l",
            b=batch_size,
            l=sequence_length,
        )
        dt_y = rearrange(
            dt_y,
            "d (b l) -> b d l",
            b=batch_size,
            l=sequence_length,
        )

        A_x = -torch.exp(self.A_log_x.float())
        A_y = -torch.exp(self.A_log_y.float())
        B_x = rearrange(
            B_x,
            "(b l) n -> b n l",
            b=batch_size,
            l=sequence_length,
        ).contiguous()
        B_y = rearrange(
            B_y,
            "(b l) n -> b n l",
            b=batch_size,
            l=sequence_length,
        ).contiguous()
        C_x = rearrange(
            C_x,
            "(b l) n -> b n l",
            b=batch_size,
            l=sequence_length,
        ).contiguous()
        C_y = rearrange(
            C_y,
            "(b l) n -> b n l",
            b=batch_size,
            l=sequence_length,
        ).contiguous()

        # Mutual cross-modal interaction:
        # X uses C_y, while Y uses C_x.
        x_out = selective_scan_fn(
            x,
            dt_x,
            A_x,
            B_x,
            C_y,
            self.D_x.float(),
            delta_bias=self.dt_proj_x.bias.float(),
            delta_softplus=True,
        )
        y_out = selective_scan_fn(
            y,
            dt_y,
            A_y,
            B_y,
            C_x,
            self.D_y.float(),
            delta_bias=self.dt_proj_y.bias.float(),
            delta_softplus=True,
        )

        x_out = self.out_proj_x(
            rearrange(x_out, "b d l -> b l d")
        )
        y_out = self.out_proj_y(
            rearrange(y_out, "b d l -> b l d")
        )
        return x_out + residual_x, y_out + residual_y
