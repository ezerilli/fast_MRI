import fastmri
import os
import h5py
import numpy as np
import torch
from pathlib import Path
import logging
from fastmri.data import transforms as fastmri_transforms
from typing import Callable, Optional, Tuple, Union
import pickle


class KspaceVolume:

    def __init__(self, file: Union[str, Path], pytorch_device: Optional[str] = None, example_number=None, metadata=None,
                 recons_key=None):
        self._data_file = file
        if pytorch_device:
            self._pytorch_device = pytorch_device
        elif torch.cuda.is_available():
            self._pytorch_device = 'cuda'
        else:
            self._pytorch_device = 'cpu'

        self.example_number = example_number
        self.metadata = metadata
        self._recons_key = recons_key

        self._shape = None
        self._size = None

        self._kspace_array = None
        self._kspace_raw_tensor = None
        self._complex_images = None
        self._real_images = None

        self._mask = None
        self._targets = None
        self._attrs = None

    def _read_h5(self):
        with h5py.File(self._data_file, "r") as hf:
            self._mask = np.asarray(hf["mask"]) if "mask" in hf else None
            self._kspace_array = hf['kspace'][()]
            self._targets = hf[self._recons_key] if self._recons_key and self._recons_key in hf else None
            self._attrs = dict(hf.attrs)
            self._attrs.update(self.metadata)

    @property
    def attrs(self):
        if self._attrs is None:
            self._read_h5()
        return self._attrs

    @property
    def complex_images(self) -> torch.Tensor:
        if self._complex_images is None:
            # Apply Inverse Fourier Transform to raw kspace tensor to get the complex images
            self._complex_images = fastmri.ifft2c(self.kspace_raw_tensor)
            # After setting, None-out the other representations to conserve memory
            self._kspace_raw_tensor = None
            self._kspace_array = None
            self._real_images = None
        return self._complex_images

    @property
    def data_file(self) -> str:
        return self._data_file

    @property
    def kspace_array(self) -> np.ndarray:
        if self._kspace_array is None:
            self._read_h5()
            # After setting, None-out the other representations to conserve memory
            self._kspace_raw_tensor = None
            self._complex_images = None
            self._real_images = None
        return self._kspace_array

    @property
    def kspace_raw_tensor(self) -> torch.Tensor:
        if self._kspace_raw_tensor is None:
            self._kspace_raw_tensor = fastmri_transforms.to_tensor(np.expand_dims(self.kspace_array, axis=1))
            # After setting, None-out the other representations to conserve memory
            self._kspace_array = None
            self._complex_images = None
            self._real_images = None
        return self._kspace_raw_tensor

    @property
    def mask(self):
        if self._mask is None:
            self._read_h5()
        return self._mask

    @property
    def num_slices(self) -> int:
        return self.shape[0]

    @property
    def real_images(self) -> torch.Tensor:
        if self._real_images is None:
            # Compute absolute value of complex images tensor to get a real images
            # TODO: may need to adjust in some contexts to not need separate absolute value tensor
            self._real_images = fastmri.complex_abs(self.complex_images)
            # After setting, None-out the other representations to conserve memory
            self._kspace_array = None
            self._kspace_raw_tensor = None
            self._complex_images = None
        return self._real_images

    def release(self):
        """
        Release the data in memory.
        """
        self._kspace_array = None
        self._kspace_raw_tensor = None
        self._complex_images = None
        self._real_images = None

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._shape is None:
            if self._kspace_array is not None:
                self._shape = self._kspace_array.shape
            elif self._kspace_raw_tensor is not None:
                self._shape = self._kspace_raw_tensor.shape
            elif self._complex_images is not None:
                self._shape = self._complex_images.shape
            elif self._real_images is not None:
                self._shape = self._real_images.shape
            else:
                self._shape = self.kspace_array.shape
        return self._shape

    @property
    def targets(self):
        if self._targets is None:
            self._read_h5()
        return self._targets


class HeldOutSslKspaceVolume(KspaceVolume):
    def __init__(self, file: Union[str, Path], theta_lambda_ratio: float = 0.5, pytorch_device: Optional[str] = None,
                 example_number=None, metadata=None):
        super(HeldOutSslKspaceVolume, self).__init__(file, pytorch_device, example_number, metadata)
        if theta_lambda_ratio >= 1.0 or theta_lambda_ratio <= 0:
            raise ValueError('Invalid ratio value for hold out volume slice subset')

        self.theta_lambda_ratio = theta_lambda_ratio
        self._theta_indices = None
        self._lambda_indices = None

    @property
    def lambda_indices(self):
        if self._lambda_indices is None:
            arange = np.arange(self.num_slices)
            self._lambda_indices = arange[np.isin(arange, self._theta_indices, invert=True)]
        return self._lambda_indices

    @property
    def lambda_subset(self) -> torch.Tensor:
        return self.kspace_raw_tensor[self._lambda_indices]

    @property
    def theta_indices(self):
        if self._theta_indices is None:
            # TODO: come back and implement overlap
            arange = np.arange(self.num_slices)
            self._theta_indices = np.random.choice(arange, size=int(self.num_slices * self.theta_lambda_ratio),
                                                   replace=False)
        return self._theta_indices

    @property
    def theta_subset(self) -> torch.Tensor:
        return self.kspace_raw_tensor[self.theta_indices]


class KspaceVolumeDataset(fastmri.data.SliceDataset):

    def __init__(
            self,
            root: Union[str, Path, os.PathLike],
            challenge: str,
            transform: Optional[Callable] = None,
            use_dataset_cache: bool = False,
            volume_sample_rate: Optional[float] = None,
            dataset_cache_file: Union[str, Path, os.PathLike] = "ssl_dataset_cache.pkl",
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        # set sampling mode
        sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            ex_num = 0
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
                example = (fname, ex_num, metadata)
                self.examples.append(example)
                ex_num += 1

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if volume_sample_rate < 1.0:  # sample by volume
            sample_size = int(len(self.examples) * volume_sample_rate)
            arange = np.arange(len(self.examples))
            subsample_indices = np.sort(np.random.Generator.choice(arange, size=sample_size, replace=False))
            subsample = list()
            for i in range(subsample_indices.size):
                indx = subsample_indices[i]
                subsample.append(self.examples[indx])
            self.examples = subsample

    def __getitem__(self, i: int):
        fname, ex_num, metadata = self.examples[i]
        volume = HeldOutSslKspaceVolume(file=fname, example_number=ex_num, metadata=metadata)
        if self.transform is None:
            theta_images = volume.real_images[volume.theta_indices]
            lambda_images = volume.real_images[volume.lambda_indices]
            max_value = volume.attrs["max"] if "max" in volume.attrs.keys() else 0.0
            sample = (volume.kspace_raw_tensor, theta_images, lambda_images, None, None, volume.attrs, fname.name, max_value)
        else:
            # vol_raw_kspace, theta_images, lambda_images, theta_mean, theta_std, volume.attrs, volume.data_file, max_value
            sample = self.transform(volume)

        return sample
