# from torch import nn
# from typing import Callable
# import torch

# # Adapted from : https://github.com/Project-MONAI/MONAI/blob/46a5272196a6c2590ca2589029eed8e4d56ff008/monai/losses/tversky.py#L24-L162

# class TverskyLoss(_Loss):
#     """
#     Compute the Tversky loss defined in:

#         Sadegh et al. (2017) Tversky loss function for image segmentation
#         using 3D fully convolutional deep networks. (https://arxiv.org/abs/1706.05721)

#     Adapted from:
#         https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L631

#     """

#     def __init__(
#         self,
#         include_background: bool = True,
#         to_onehot_y: bool = False,
#         sigmoid: bool = False,
#         softmax: bool = False,
#         other_act: Callable | None = None,
#         alpha: float = 0.5,
#         beta: float = 0.5,
#         reduction: LossReduction | str = LossReduction.MEAN,
#         smooth_nr: float = 1e-5,
#         smooth_dr: float = 1e-5,
#         batch: bool = False,
#     ) -> None:
#         """
#         Args:
#             include_background: If False channel index 0 (background category) is excluded from the calculation.
#             to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
#             sigmoid: If True, apply a sigmoid function to the prediction.
#             softmax: If True, apply a softmax function to the prediction.
#             other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
#                 other activation layers, Defaults to ``None``. for example:
#                 `other_act = torch.tanh`.
#             alpha: weight of false positives
#             beta: weight of false negatives
#             reduction: {``"none"``, ``"mean"``, ``"sum"``}
#                 Specifies the reduction to apply to the output. Defaults to ``"mean"``.

#                 - ``"none"``: no reduction will be applied.
#                 - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
#                 - ``"sum"``: the output will be summed.

#             smooth_nr: a small constant added to the numerator to avoid zero.
#             smooth_dr: a small constant added to the denominator to avoid nan.
#             batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
#                 Defaults to False, a Dice loss value is computed independently from each item in the batch
#                 before any `reduction`.

#         Raises:
#             TypeError: When ``other_act`` is not an ``Optional[Callable]``.
#             ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
#                 Incompatible values.

#         """

#         super().__init__(reduction=LossReduction(reduction).value)
#         if other_act is not None and not callable(other_act):
#             raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
#         if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
#             raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
#         self.include_background = include_background
#         self.to_onehot_y = to_onehot_y
#         self.sigmoid = sigmoid
#         self.softmax = softmax
#         self.other_act = other_act
#         self.alpha = alpha
#         self.beta = beta
#         self.smooth_nr = float(smooth_nr)
#         self.smooth_dr = float(smooth_dr)
#         self.batch = batch

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             input: the shape should be BNH[WD].
#             target: the shape should be BNH[WD].

#         Raises:
#             ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

#         """
#         if self.sigmoid:
#             input = torch.sigmoid(input)

#         n_pred_ch = input.shape[1]
#         if self.softmax:
#             if n_pred_ch == 1:
#                 warnings.warn("single channel prediction, `softmax=True` ignored.")
#             else:
#                 input = torch.softmax(input, 1)

#         if self.other_act is not None:
#             input = self.other_act(input)

#         if self.to_onehot_y:
#             if n_pred_ch == 1:
#                 warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
#             else:
#                 target = one_hot(target, num_classes=n_pred_ch)

#         if not self.include_background:
#             if n_pred_ch == 1:
#                 warnings.warn("single channel prediction, `include_background=False` ignored.")
#             else:
#                 # if skipping background, removing first channel
#                 target = target[:, 1:]
#                 input = input[:, 1:]

#         if target.shape != input.shape:
#             raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

#         p0 = input
#         p1 = 1 - p0
#         g0 = target
#         g1 = 1 - g0

#         # reducing only spatial dimensions (not batch nor channels)
#         reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
#         if self.batch:
#             # reducing spatial dimensions and batch
#             reduce_axis = [0] + reduce_axis

#         tp = torch.sum(p0 * g0, reduce_axis)
#         fp = self.alpha * torch.sum(p0 * g1, reduce_axis)
#         fn = self.beta * torch.sum(p1 * g0, reduce_axis)
#         numerator = tp + self.smooth_nr
#         denominator = tp + fp + fn + self.smooth_dr

#         score: torch.Tensor = 1.0 - numerator / denominator

#         if self.reduction == LossReduction.SUM.value:
#             return torch.sum(score)  # sum over the batch and channel dims
#         if self.reduction == LossReduction.NONE.value:
#             return score  # returns [N, num_classes] losses
#         if self.reduction == LossReduction.MEAN.value:
#             return torch.mean(score)
#         raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')