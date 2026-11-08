from __future__ import annotations

import torch
from mamba_ssm import Mamba
from torch import nn

from .function_last import Cross_Modal_Mamba
from .kan import KANModalEncoder


class MLPClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class KBM(nn.Module):
    """KAN modality encoders with forward, reverse, and cross-modal Mamba."""

    def __init__(
        self,
        num_classes: int,
        feature_list,
        hidden: int = 32,
        dim_embedding: int = 32,
        d_state: int = 64,
    ):
        super().__init__()
        if len(feature_list) < 2:
            raise ValueError("Cross-modal Mamba requires at least two modalities.")

        self.num_modal = len(feature_list)
        self.dim_embedding = dim_embedding
        self.kan_encoder = KANModalEncoder(
            feature_list,
            hidden_dim=hidden,
            embed_dim=dim_embedding,
        )
        self.modal_heads = nn.ModuleList(
            [
                MLPClassifier(dim_embedding, hidden, num_classes)
                for _ in range(self.num_modal)
            ]
        )
        self.ssm_encoder = MultimodalStateSpaceNetworks(
            num_modal=self.num_modal,
            dim_embedding=dim_embedding,
            d_state=d_state,
        )
        self.fusion_head = MLPClassifier(
            input_size=dim_embedding * self.num_modal,
            hidden_size=hidden,
            num_classes=num_classes,
        )

    def forward(self, features: torch.Tensor):
        modal_embeddings = self.kan_encoder(features)
        modal_logits = [
            head(embedding)
            for head, embedding in zip(
                self.modal_heads,
                modal_embeddings.unbind(dim=1),
            )
        ]

        fused_embeddings = self.ssm_encoder(modal_embeddings)
        fusion_logits = self.fusion_head(
            fused_embeddings.flatten(start_dim=1)
        )
        return (fusion_logits, *modal_logits)


class MultimodalStateSpaceNetworks(nn.Module):
    def __init__(
        self,
        num_modal: int,
        dim_embedding: int = 32,
        d_state: int = 64,
    ):
        super().__init__()
        self.modal_embedding = nn.Parameter(
            torch.zeros(1, num_modal, dim_embedding)
        )
        nn.init.trunc_normal_(self.modal_embedding, std=0.02)
        self.forward_norm = nn.LayerNorm(dim_embedding)
        self.reverse_norm = nn.LayerNorm(dim_embedding)
        self.forward_mamba = Mamba(
            d_model=dim_embedding,
            d_state=d_state,
            d_conv=2,
            expand=2,
        )
        self.reverse_mamba = Mamba(
            d_model=dim_embedding,
            d_state=d_state,
            d_conv=2,
            expand=2,
        )
        self.cross_modal_mamba = Cross_Modal_Mamba(
            d_model=dim_embedding,
            d_state=d_state,
            ssm_ratio=2,
        )
        self.direction_gate = nn.Sequential(
            nn.Linear(dim_embedding * 2, dim_embedding),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(dim_embedding)
    def forward(self, modality_embeddings: torch.Tensor) -> torch.Tensor:
        if modality_embeddings.ndim != 3:
            raise ValueError(
                "Expected modality embeddings with shape [batch, modalities, dim]."
            )
        tokens = modality_embeddings + self.modal_embedding
        forward_input = self.forward_norm(tokens)
        forward_output = self.forward_mamba(forward_input)
        forward_output = forward_output + modality_embeddings

        reverse_input = self.reverse_norm(modality_embeddings)
        reverse_input = torch.flip(reverse_input, dims=[1])
        reverse_output = self.reverse_mamba(reverse_input)
        reverse_output = torch.flip(reverse_output, dims=[1])
        reverse_output = reverse_output + modality_embeddings

        cross_forward, cross_reverse = self.cross_modal_mamba(
            forward_output,
            reverse_output,
        )
        
        gate = self.direction_gate(
            torch.cat([cross_forward, cross_reverse], dim=-1)
        )

        fused = gate * cross_forward + (1.0 - gate) * cross_reverse
        return self.output_norm(fused)

