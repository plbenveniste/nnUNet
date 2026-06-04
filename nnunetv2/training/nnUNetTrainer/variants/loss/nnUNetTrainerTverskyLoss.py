import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class TverskyLoss(nn.Module):
    """Tversky loss where alpha penalises false negatives and beta penalises false positives.
    alpha=0.3, beta=0.7 increases recall, useful for small lesion segmentation."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5,
                 batch_dice: bool = False, do_bg: bool = False, ddp: bool = False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.ddp = ddp

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, loss_mask=None):
        p = softmax_helper_dim1(net_output)
        shp = net_output.shape
        target_onehot = torch.zeros(shp, device=net_output.device, dtype=p.dtype)
        target_onehot.scatter_(1, target.long(), 1)

        axes = list(range(2, len(shp)))
        if self.batch_dice:
            axes = [0] + axes

        if not self.do_bg:
            p = p[:, 1:]
            target_onehot = target_onehot[:, 1:]

        if loss_mask is not None:
            p = p * loss_mask
            target_onehot = target_onehot * loss_mask

        tp = (p * target_onehot).sum(dim=axes)
        fp = (p * (1 - target_onehot)).sum(dim=axes)
        fn = ((1 - p) * target_onehot).sum(dim=axes)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - tversky_index).mean()


class DC_and_Tversky_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, tversky_kwargs, weight_tversky=1, weight_dice=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_tversky = weight_tversky
        self.ignore_label = ignore_label

        self.tversky = TverskyLoss(**tversky_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        tversky_loss = self.tversky(net_output, target_dice, loss_mask=mask) \
            if self.weight_tversky != 0 and (self.ignore_label is None or num_fg > 0) else 0

        return self.weight_tversky * tversky_loss + self.weight_dice * dc_loss


class nnUNetTrainerTverskyLoss(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("nnUNetTrainerTverskyLoss does not support region-based labels")

        loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-5,
                           batch_dice=self.configuration_manager.batch_dice,
                           do_bg=False, ddp=self.is_ddp)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerTverskyLoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("nnUNetTrainerTverskyLoss_noSmooth does not support region-based labels")

        loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=0,
                           batch_dice=self.configuration_manager.batch_dice,
                           do_bg=False, ddp=self.is_ddp)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDiceTverskyLoss(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("nnUNetTrainerDiceTverskyLoss does not support region-based labels")

        loss = DC_and_Tversky_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                              'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            tversky_kwargs={'alpha': 0.3, 'beta': 0.7, 'smooth': 1e-5,
                            'batch_dice': self.configuration_manager.batch_dice,
                            'do_bg': False, 'ddp': self.is_ddp},
            weight_tversky=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDiceTverskyLoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("nnUNetTrainerDiceTverskyLoss_noSmooth does not support region-based labels")

        loss = DC_and_Tversky_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                              'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp},
            tversky_kwargs={'alpha': 0.3, 'beta': 0.7, 'smooth': 0,
                            'batch_dice': self.configuration_manager.batch_dice,
                            'do_bg': False, 'ddp': self.is_ddp},
            weight_tversky=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
