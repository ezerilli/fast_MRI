"""
@ezerilli

Module for wrapping a U-net for DINO (self-distillation with no labels).
"""

import torch
from .unet import Unet
from torch import nn
from torch.nn import functional as F


class DinoNet(nn.Module):
    """
    PyTorch implementation of a Dino U-Net model.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            chans: int = 32,
            num_pool_layers: int = 4,
            drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()
        self.model = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
        )

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crops:  list of Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        if not isinstance(crops, list):
            crops = [crops]

        # TODO: is that really necessary? We can have multi-crops in image reconstruction
        crops_sizes = torch.tensor([crop.shape[-1] for crop in crops])
        sizes_count = torch.unique_consecutive(crops_sizes, return_counts=True)[1]
        cat_splits = torch.cumsum(sizes_count, dim=0)

        split_start = 0
        for split_end in cat_splits:
            model_output = self.model(torch.cat(crops[split_start: split_end], dim=0))
            output = model_output if split_start == 0 else torch.cat([output, model_output], dim=0)
            split_start = split_end

        return output


class DinoLoss(nn.Module):

    @staticmethod
    def forward(student_output, teacher_output):
        """
        F1-loss between outputs of the teacher and student networks.
        """
        # Teacher detaching
        for i, output in enumerate(teacher_output):
            teacher_output[i] = output.detach()

        total_loss = 0
        n_loss_terms = 0
        for i, teacher_view in enumerate(teacher_output):
            for j, student_view in enumerate(student_output):
                if i == j:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = F.l1_loss(student_view, teacher_view)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        return total_loss
