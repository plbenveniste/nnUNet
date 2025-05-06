from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import nnUNetTrainer_DASegOrd0_NoMirroring
from typing import Tuple, Union, List
import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from datetime import datetime
import wandb 
from torch import autocast 
import os
from nnunetv2.utilities.helpers import empty_cache, dummy_context
import torch
from nnunetv2.training.nnUNetTrainer.transforms.transforms import ConvTransform, HistogramEqualTransform, FunctionTransform, ImageFromSegTransform, RedistributeTransform, ArtifactTransform, SpatialCustomTransform
from nnunetv2.training.nnUNetTrainer.transforms.transforms_multimodal import RedistributeTransformMultimodal
import matplotlib.pyplot as plt 


class nnUNetTrainerDAExt_multimodal(nnUNetTrainer):
    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            retain_stats: bool = False
    ) -> BasicTransform:
        transforms = []

        ### Adds custom nnunet transforms
        ## Contrast transforms
        # Scharr filter
        transforms.append(RandomTransform(
            ConvTransform(
                kernel_type='Scharr', 
                absolute=True, 
                retain_stats=retain_stats
            ), apply_probability=0.15
        ))
        
        # Apply functions
        func_list = [
            lambda x:torch.log(1 + x), # Log
            torch.sqrt, # sqrt
            torch.sin, # sin
            torch.exp, # exp
            lambda x:1/(1 + torch.exp(-x)), # sig
        ]

        for func in func_list:
            transforms.append(RandomTransform(
                FunctionTransform(
                    function=func,
                    retain_stats=retain_stats
                ), apply_probability=0.05
            ))

        # Histogram equalization
        transforms.append(RandomTransform(
            HistogramEqualTransform(
                retain_stats=retain_stats
            ), apply_probability=0.1
        ))

        # Image from segmentation
        transforms.append(RandomTransform(
            ImageFromSegTransform(
                retain_stats=retain_stats
            ), apply_probability=0
        ))
        
        # Redistribute segmentation values
        transforms.append(RandomTransform(
            RedistributeTransformMultimodal(
                retain_stats=retain_stats
            ), apply_probability=0.5
        ))
        
        ## Artifacts generation
        # Motion, Ghosting, Spike, Bias field, Blur, Noise, Swap
        transforms.append(RandomTransform(
            ArtifactTransform(
                motion=True,
                ghosting=True,
                spike=True,
                bias_field=True,
                blur=True,
                noise=True,
                swap=False,
                random_pick=True
            ), apply_probability=0.7
        ))

        ## Spatial transforms
        # Flip, Affine, Elastic, Anisotropy
        transforms.append(RandomTransform(
            SpatialCustomTransform(
                flip=True,
                affine=True,
                elastic=True,
                anisotropy=True,
                random_pick=True
            ), apply_probability=0.6
        ))
        ### End of customs

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainerDAExt_DiceCELoss_noSmooth_multimodal(nnUNetTrainerDAExt_multimodal):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    

# class nnUNetTrainerDAExt_DiceCELoss_noSmooth_wandB(nnUNetTrainerDAExt):

#     def run_training(self):
#         self.on_train_start()
#         output_path = os.path.join("output_path", str(datetime.now().date()) +"_" +str(datetime.now().time()))
#         os.makedirs(output_path, exist_ok=True)

#         wandb.init(project=f'you_project_name',  dir=output_path)

#         for epoch in range(self.current_epoch, self.num_epochs):
#             self.on_epoch_start()

#             self.on_train_epoch_start()
#             train_outputs = []
#             for batch_id in range(self.num_iterations_per_epoch):
#                 train_outputs.append(self.train_step(next(self.dataloader_train),batch_id))
#             self.on_train_epoch_end(train_outputs)

#             with torch.no_grad():
#                 self.on_validation_epoch_start()
#                 val_outputs = []
#                 for batch_id in range(self.num_val_iterations_per_epoch):
#                     val_outputs.append(self.validation_step(next(self.dataloader_val)))
#                 self.on_validation_epoch_end(val_outputs)

#             self.on_epoch_end()
#         wandb.finish()  
#         self.on_train_end()

#     def train_step(self, batch: dict, batch_id: int) -> dict:
#         data = batch['data']
#         target = batch['target']

#         data = data.to(self.device, non_blocking=True)
#         if isinstance(target, list):
#             target = [i.to(self.device, non_blocking=True) for i in target]
#         else:
#             target = target.to(self.device, non_blocking=True)

#         self.optimizer.zero_grad(set_to_none=True)
#         # Autocast can be annoying
#         # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
#         # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
#         # So autocast will only be active if we have a cuda device.
#         with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
#             output = self.network(data)
#             # del data
#             l = self.loss(output, target)

#             if batch_id == 0: 
#                 train_image= data[0].detach().cpu().squeeze().float().numpy()
#                 train_gt= target[0].detach().cpu().squeeze().float().numpy()[0]
#                 train_pred = np.argmax(output[0].detach().cpu().squeeze().numpy(), axis=1)[0]
                

#                 fig = plot_slices_combined(combined=train_image,
#                             gt=train_gt,
#                             pred=train_pred,
#                                     )

#                 wandb.log({"training images": wandb.Image(fig)})
#                 plt.close(fig)

#         if self.grad_scaler is not None:
#             self.grad_scaler.scale(l).backward()
#             self.grad_scaler.unscale_(self.optimizer)
#             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
#             self.grad_scaler.step(self.optimizer)
#             self.grad_scaler.update()
#         else:
#             l.backward()
#             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
#             self.optimizer.step()
#         return {'loss': l.detach().cpu().numpy()}

#This function create a plot that we will send to WandB 

def plot_slices_combined(combined, gt, pred, debug=False):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    
    mid_sagittal = combined.shape[2]//2
    
    
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(3, 6, figsize=(10, 6))
    fig.suptitle('T2 Image --> Other contrast --> Ground Truth --> Prediction')
    if np.all(combined == 0):
        print("Array contains only zeros")
    for i in range(6):
        axs[0, i].imshow(combined[:,:,mid_sagittal-3+i].T, cmap='gray'); axs[0, i].axis('off') 
        axs[1, i].imshow(gt[:,:,mid_sagittal-3+i].T); axs[1, i].axis('off')
        axs[2, i].imshow(pred[:,:,mid_sagittal-3+i].T); axs[2, i].axis('off')
        
    
    plt.tight_layout()
    fig.show()
    return fig


class nnUNetTrainerDAExt_DiceCELoss_noSmooth_multimodal(nnUNetTrainerDAExt_multimodal):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    

class nnUNetTrainerDAExt_DiceCELoss_noSmooth_300epochs_multimodal(nnUNetTrainerDAExt_DiceCELoss_noSmooth_multimodal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
                     super().__init__(plans, configuration, fold, dataset_json, device)
                     self.num_epochs = 300



class nnUNetTrainerDAExt_DiceCELoss_noSmooth_300epochs_wandb_multimodal(nnUNetTrainerDAExt_DiceCELoss_noSmooth_300epochs_multimodal):

    def run_training(self):
        self.on_train_start()
        output_path = os.path.join("output_path", str(datetime.now().date()) +"_" +str(datetime.now().time()))
        os.makedirs(output_path, exist_ok=True)

        wandb.init(project=f'ms-seg-nnunet',  dir=output_path)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train),batch_id))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
        wandb.finish()  
        self.on_train_end()

    def train_step(self, batch: dict, batch_id: int) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

            if batch_id == 0: 
                train_image= data[0].detach().cpu().squeeze().float().numpy()
                train_gt= target[0].detach().cpu().squeeze().float().numpy()[0]
                train_pred = np.argmax(output[0].detach().cpu().squeeze().numpy(), axis=1)[0]
                

                fig = plot_slices_combined(combined=train_image,
                            gt=train_gt,
                            pred=train_pred,
                                    )

                wandb.log({"training images": wandb.Image(fig)})
                plt.close(fig)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

class nnUNetTrainerDAExt_DiceCELoss_noSmooth_1000epochs_multimodal(nnUNetTrainerDAExt_DiceCELoss_noSmooth_multimodal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
                     super().__init__(plans, configuration, fold, dataset_json, device)
                     self.num_epochs = 1000


class nnUNetTrainerDAExt_DiceCELoss_noSmooth_2000epochs_multimodal(nnUNetTrainerDAExt_DiceCELoss_noSmooth_multimodal):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
                     super().__init__(plans, configuration, fold, dataset_json, device)
                     self.num_epochs = 2000

