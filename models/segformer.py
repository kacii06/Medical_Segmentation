"""SegFormer-B0 for binary segmentation, initialized from scratch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerB0Scratch(nn.Module):
    """A randomly initialized SegFormer-B0 with a one-channel output."""

    def __init__(self, num_labels: int = 1):
        super().__init__()

        config = SegformerConfig(
            num_channels=3,
            num_labels=num_labels,
            hidden_sizes=[32, 64, 160, 256],
            depths=[2, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            sr_ratios=[8, 4, 2, 1],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4],
            decoder_hidden_size=256,
        )

        # Constructing from config initializes every parameter randomly.
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits

        # SegFormer produces lower-resolution logits. Return full-resolution
        # logits so the existing loss and metrics can be reused unchanged.
        if logits.shape[-2:] != pixel_values.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_segformer(num_labels: int = 1) -> SegFormerB0Scratch:
    """Factory for the scratch SegFormer-B0 baseline."""
    return SegFormerB0Scratch(num_labels=num_labels)
