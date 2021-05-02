import fastmri
import numpy as np
import torch
from fastmri.data import transforms as fastmri_transforms
from fastmri.data.subsample import MaskFunc
from typing import Dict, Optional, Tuple


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

    def __call__(self, kspace: np.ndarray, mask: np.ndarray, target: np.ndarray, attrs: Dict, fname: str,
                 slice_num: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
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
        kspace = fastmri_transforms.to_tensor(kspace)
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = fastmri_transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = fastmri_transforms.complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)

        # normalize input
        image, mean, std = fastmri_transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target = fastmri_transforms.to_tensor(target)
            target = fastmri_transforms.center_crop(target, crop_size)
            target = fastmri_transforms.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, mean, std, fname, slice_num, max_value
