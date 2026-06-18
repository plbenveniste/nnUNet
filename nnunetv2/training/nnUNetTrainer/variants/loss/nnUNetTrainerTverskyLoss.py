import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import TverskyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerTverskyLoss(nnUNetTrainer):
    """Trainer using Tversky loss with alpha=0.3, beta=0.7 (penalises FN more than FP)."""

    def _build_loss(self):
        loss = TverskyLoss(
            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1,
            batch_dice=self.configuration_manager.batch_dice,
            do_bg=self.label_manager.has_regions,
            smooth=1e-5,
            ddp=self.is_ddp,
            alpha=0.3,
            beta=0.7,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainerTverskyLoss_1000epochs(nnUNetTrainerTverskyLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000


class nnUNetTrainerTverskyLoss_2000epochs(nnUNetTrainerTverskyLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000


class nnUNetTrainerTverskyLoss_4000epochs(nnUNetTrainerTverskyLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 4000
