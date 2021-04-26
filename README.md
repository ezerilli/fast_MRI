# Fast MRI
https://github.com/ezerilli/fast_MRI

### SETTING UP THE ENVIRONMENT 👨🏻‍💻👨🏻‍💻👨🏻‍💻

The following steps lead to setup the working environment for the fast MRI. 👨🏻‍💻‍📚‍‍‍‍

Installing the conda environment is a ready-to-use solution to be able to run python scripts without having to worry 
about the packages and versions used. Alternatively, you can install each of the packages in `environment.yaml` on your 
own independently with pip or conda.

1. Start by installing Anaconda for your operating system following the instructions [here](https://docs.anaconda.com/anaconda/install/).

2. Now install the environment described in `environment.yaml`:
```bash
conda env create -f environment.yaml
```

4. To activate the environment run:
```bash
conda activate fast_mri
```

5. To deactivate the environment run:
```bash
conda deactivate
```

6. To update it run:
```bash
conda env update --prefix ./env --file environment.yaml  --prune
```

### FAST-MRI ‍🔥🔥🔥

####  Motivation

Magnetic resonance imaging (MRI) scans are one of the most powerful imaging modalities for medical image diagnosis due to their adaptability 
and unparalleled soft tissue contrast. However, in contrast to imaging techniques such as x-ray CT and ultrasound, MRI scans have long 
acquisition times, with protocols often taking as long as an hour. The long scan time of MRI is a primary driver of the large monetary cost for MRI examinations. As such, shortening MRI examinations - while maintaining its high image quality and soft tissue contrast - is a topic of significant interest to the medical community. Decreasing the length of scans could decrease the cost, which would broaden access and may even allow MRI’s use in new applications where other imaging modalities are the current standard.
For MR imaging accelerations, deep neural networks (DNNs) are the current state-of-the-art. DNNs are typically trained in a supervised 
fashion with metrics such as mean-squared error, but acquiring ground-truth fully-sampled data for supervised learning can be prohibitively 
expensive. A more sustainable approach for training MRI models would be to use self-supervision on already-subsampled data.

#### Approaches

The reason for MRI scan length is the need to sample spatial frequencies of the object. The object’s spatial frequency representation - called “k-space” - must be fully sampled into order to avoid classical Shannon-Nyquist aliasing. The field of MRI reconstruction has developed a number of techniques for avoiding the aliasing while reducing sampling. The most common technique (and only technique with current clinical adoption) is to use parallel receive channels [1, 2, 3]. In this setting, the linear inverse problem becomes overdetermined even with Fourier undersampling, and an image can be reconstructed. Parallel imaging can be further improved with regularized approaches like compressed sensing, but compressed sensing uses handcrafted mathematical features of images for optimization that don’t represent the features of the actual data.
Due to this failing, more recently deep learning methods have supplanted compressed sensing [4, 5]. Deep learning methods learn more natural image features, typically encoding them as weights in deep neural networks (DNNs). The simplest deep learning models applied to image reconstruction include U-Nets and ResNets. However, these image-to-image models neglect a lot of other information that can be leveraged via MRI physics equations. Hybrid approaches involving parallel imaging data models with DNN-based priors are the current state-of-the-art methods for accelerated MR image reconstruction, and a number of such approaches are currently being used for adoption in the clinic.
Metrics
The largest data set for MR image reconstruction is the fastMRI data set released by Facebook AI Research and NYU Langone Health 
(https://fastmri.med.nyu.edu/), which includes data for both knee and brain imaging. Other data sets include the Calgary-Campinas Public Brain 
MR Dataset (https://sites.google.com/view/calgary-campinas-dataset/home) and mridata.org (http://mridata.org/). An important aspect of these 
data sets over other data sets is their inclusion of raw k-space data. Many medical imaging data sets only release DICOM image- format data, 
which discards most of the information in the raw k-space and warps the image statistics. Deep learning models trained at least in part on 
raw k-space data are the only acceptable models for MR image reconstruction.
Metrics for training MR reconstruction methods include standard reference distance metrics such as mean-squared error (MSE) and structural 
similarity (SSIM). However, these metrics don’t exactly correlate with clinical quality, so the final validation step typically involves a human reader study with certified radiologists.

#### Scope

These methods are typically trained via supervised learning on data that has the full k-space sampled. In practice most MRI data is 
already subsampled via parallel imaging [1, 2, 3]. Acquiring the full k-space is prohibitively expensive. Methods have been proposed based 
on self-supervision by holding out samples of subsampled acquisitions for training [6], but these methods have yet to be evaluated on large 
data sets. It is also unknown whether there are self- supervision techniques other than hold-out sampling that might be better for performance. 
A successful project from this proposal might include the following components:

1. Assess performance of self-supervised hold-out methods on large data sets, such as the fastMRI data set.
2. Examine other possible self-supervision techniques.
3. Given results from (1), what recommendations can be made on self-supervised training methods.

#### Resources

A basic primer on MRI can be found in the fastMRI arXiv paper: https://arxiv.org/abs/1811.08839. 
For training, students can use the knee portion of the fastMRI data set, available at https://fastmri.med.nyu.edu/. 
To make prototyping easier, the fastMRI data set includes single-coil as well as multicoil data. Baseline U-Net models for fastMRI data 
are implemented in the fastMRI repository at https://github.com/facebookresearch/fastMRI. Training of both the baseline U-Net as well as 
state-of-the-art unrolled models typically requires 3 days of training on an Nvidia GPU with 16 GB of memory.


### PROPOSAL


#### Team: 

Learner 

#### Facebook Project: 

Yes 

#### Project Title: 

fastMRI 

#### Project Summary: 

Magnetic resonance imaging (MRI) scans are one of the most powerful imaging modalities for medical image diagnosis due to their adaptability 
and unparalleled soft tissue contrast. The cost of MRI being high is due to extended acquisition times required by the procedure, which can 
reach up to 1 hours. As such, shortening MRI examinations ( fastMRI ) - while maintaining its high image quality and soft tissue contrast - 
is a topic of significant interest to the medical community. Decreasing MRI scan time would reduce the cost and also increase the scope of 
applying MRI use in areas where imaging modalities are used due to cost and time limitations of MRI. Deep neural networks (DNNs) are a 
potential technique by which fastMRI can be achieved without compromising on the quality of the Image. DNNs are supervised Machine Learning 
techniques with a more sustainable approach for training MRI models would be to use self-supervision on already-subsampled data. 

#### Approach: 

1. Replicate the SSL approach presented in [6].  
2. Using the paper [8] and fastMRI dataset [9] reimplement the fastMRI implementation at [10] using Baseline U-Net models to scaleup for all datasets including knee and brain imaging.  
3. Analyze and replicate data reconstruction methods in [7].


### REFERENCES

- [1] Sodickson, Daniel K., and Warren J. Manning. "Simultaneous acquisition of spatial harmonics (SMASH): fast imaging with radiofrequency coil arrays." Magnetic resonance in medicine 38.4 (1997): 591-603.
- [2] Pruessmann, Klaas P., et al. "SENSE: sensitivity encoding for fast MRI." Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 42.5 (1999): 952-962.
- [3] Griswold, Mark A., et al. "Generalized autocalibrating partially parallel acquisitions (GRAPPA)." Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 47.6 (2002): 1202-1210.
- [4] Hammernik, Kerstin, et al. "Learning a variational network for reconstruction of accelerated MRI data." Magnetic resonance in medicine 79.6 (2018): 3055-3071.
- [5] Schlemper, Jo, et al. "A deep cascade of convolutional neural networks for dynamic MR image reconstruction." IEEE transactions on Medical Imaging 37.2 (2017): 491-503.
- [6] Yaman, Burhaneddin, et al. "Self‐supervised learning of physics‐guided reconstruction neural networks without fully sampled reference data." Magnetic resonance in medicine (2020) (https://arxiv.org/abs/1912.07669).
- [7] Zhang, Z., et al. “Reducing Uncertainty in Undersampled MRI Reconstruction with Active Acquisition.” (February 8, 2019 (https://arxiv.org/abs/1902.03051))  

### DATASETS

The data should be contained in the folder /data.

- [8] Knoll, F., et al. "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI" (2018) (https://arxiv.org/abs/1811.08839).
- [9] NYU Fast MRI Dataset: https://fastmri.med.nyu.edu/ 
- [10] FAIR, Fast MRI repository (https://github.com/facebookresearch/fastMRI). 

### TEAM MEMBERS

- Sudipto Lodh 
- Robert Bartel 
- Enrico Zerilli 
 
