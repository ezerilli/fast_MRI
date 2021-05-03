import fastmri
import numpy as np
import torch
from fastmri.data import transforms as fastmri_transforms
from fastmri.data.subsample import MaskFunc
from typing import Dict, Optional, Tuple
from kspace_volume import HeldOutSslKspaceVolume


class SslTransform:

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True, is_multicoil: bool = False):
        """

        Parameters
        ----------
        mask_func : Optional[MaskFunc]
            A function that can create a mask of appropriate shape.
        use_seed : bool
            If true, this class computes a pseudo random number
            generator seed from the filename. This ensures that the same
            mask is used for all the slices of a given volume every time.
        is_multicoil : bool
            Whether multicoil as opposed to single.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.which_challenge = "multicoil" if is_multicoil else "singlecoil"

    #def __call__(self, kspace: np.ndarray, mask: np.ndarray, target: np.ndarray, attrs: Dict, fname: str,
    #             slice_num: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
    def __call__(self, volume: HeldOutSslKspaceVolume):
        """

        Parameters
        ----------
        kspace
            Input k-space of shape (num_coils, rows, cols) for multi-coil data or (rows, cols) for single coil data.
        mask
            Mask from the test dataset.
        target
            Target image.
        attrs
            Acquisition related information stored in the HDF5 object.
        fname
            File name.
        slice_num
            Serial number of the slice.

        Returns
        -------
        tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        vol_raw_kspace = volume.kspace_raw_tensor
        max_value = volume.attrs["max"] if "max" in volume.attrs.keys() else 0.0
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, volume.data_file))
            ######## masked_theta_kspace, mask = fastmri_transforms.apply_mask(volume.theta_subset, self.mask_func, seed)
            subsampled_kspace, subsample_mask = fastmri_transforms.apply_mask(vol_raw_kspace, self.mask_func, seed)
            volume.subsample_mask = subsample_mask
        else:
            ######## masked_theta_kspace = volume.theta_subset
            subsampled_kspace = vol_raw_kspace
            subsample_mask = torch.ones((1, 1, 1, vol_raw_kspace.shape[-2], 1), dtype=int)
            volume.subsample_mask = subsample_mask

        # inverse Fourier transform to get zero filled solution
        ######## theta_images = fastmri.ifft2c(masked_theta_kspace)
        subsampled_images = fastmri.ifft2c(subsampled_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (volume.attrs["recon_size"][0], volume.attrs["recon_size"][1])
        ######## crop_size = (volume.attrs["recon_size"][0], volume.attrs["recon_size"][1])
        crop_size = (320, 320)
        crop_diff = int((vol_raw_kspace.shape[-2] - crop_size[0]) / 2)
        neg_crop_diff = -1 * crop_diff

        # check for FLAIR 203
        ######## if theta_images.shape[-2] < crop_size[1]:
        ########    crop_size = (theta_images.shape[-2], theta_images.shape[-2])
        if subsampled_images.shape[-2] < crop_size[1]:
            crop_size = (subsampled_images.shape[-2], subsampled_images.shape[-2])

        ######## theta_images = fastmri_transforms.complex_center_crop(theta_images, crop_size)
        ######## lambda_images = fastmri_transforms.complex_center_crop(volume.complex_images[volume.lambda_mask], crop_size)
        subsampled_images = fastmri_transforms.complex_center_crop(subsampled_images, crop_size)

        # absolute value
        ######## theta_images = fastmri.complex_abs(theta_images)
        ######## lambda_images = fastmri.complex_abs(lambda_images)
        subsampled_images = fastmri.complex_abs(subsampled_images)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            subsampled_images = fastmri.rss(subsampled_images)

        # normalize input
        normalized_out = fastmri_transforms.normalize_instance(subsampled_images, eps=1e-11)
        subsampled_images, subsample_norm_mean, subsample_norm_std = normalized_out
        subsampled_images = subsampled_images.clamp(-6, 6)

        # # normalize target
        # if target is not None:
        #     target = fastmri_transforms.to_tensor(target)
        #     target = fastmri_transforms.center_crop(target, crop_size)
        #     target = fastmri_transforms.normalize(target, theta_mean, theta_std, eps=1e-11)
        #     target = target.clamp(-6, 6)
        # else:
        #     target = torch.Tensor([0])

        return vol_raw_kspace, subsampled_kspace, subsample_mask, subsampled_images, subsample_norm_mean, \
               subsample_norm_std, volume.theta_mask[:, :, :, crop_diff:neg_crop_diff, :], \
               volume.lambda_mask[:, :, :, crop_diff:neg_crop_diff, :], volume.attrs, str(volume.data_file), max_value
