# CRISP

**A Modular Framework for Cryo-EM Image Segmentation and Processing with Conditional Random Field**

CRISP is a modular framework that facilitates experimentation with advanced image segmentation strategies for cryo-electron microscopy (cryo-EM). It streamlines the process of generating high-quality segmentation maps and integrates seamlessly with downstream workflows.

---

## Features

- **Automated Segmentation Mask Generation**: Automatically create high-quality reference segmentation maps.
- **Modular Segmentation Package**: Customize and experiment with a variety of segmentation strategies.
- **Advanced CRF Layer**: Integrates a novel Conditional Random Fields layer utilizing a regularized Frank-Wolfe algorithm and class-discriminative features to refine coarse pixel-level predictions.
- **Center finding algorithm with hyperparameter search**: Integrates several center-finding algorithms for downstream particle picking task and the best configurations is found by our proposed hyperparameter search algorithms.
---

## Manuscript

- **Title**: CRISP: A Framework for Cryo-EM Image Segmentation and Processing with Conditional Random Field  
- **Authors**: Szu-Chi Chung and Po-Cheng Chou  
- [Read the Manuscript](https://arxiv.org/abs/2502.08287)

---

## Table of Contents

- [Features](#features)
- [Manuscript](#manuscript)
- [Installation](#installation)
- [Setup](#setup)
- [Tutorials and Guides](#tutorials-and-guides)
- [Data](#data)
- [License](#license)
- [Credits](#credits)

---

## Installation

While we recommend using [Google Colab](https://colab.google/) for the best user experience, you can also install CRISP locally by following these steps:

### Prerequisites

- **Anaconda Python Distribution**  
  If you don’t have it installed, download it from [here](https://www.anaconda.com/download).

### Create a Conda Environment

```bash
conda create --name CRISP python=3.10
conda activate CRISP
```

### Install Dependencies

```bash
pip install mrcfile torch scikit-image ipython_genutils notebook
```

---

## Setup

Clone the repository and change into the project directory:

```bash
git clone https://github.com/phonchi/CryoParticleSegment.git
cd CryoParticleSegment
```

## Support components

| Component | Short Description |
| :--- | :--- |
| **Segmentation Mask Generation Pipeline** | **(Automates the creation of training masks from particle coordinates)** |
| Projection-Based Mask Generation | Generates masks by creating an initial 3D map from a few particle coordinates, reprojecting a 3D mask into 2D, and binarizing the result. This is the recommended default method. |
| Circle-Based Mask Approximation | An alternative method for lower-quality data that generates coarse masks by drawing fixed-radius circles at particle coordinates, requiring no 3D reconstruction. |
| Synthetic Dataset Simulation | Creates artificial micrographs and corresponding perfect masks from known 3D structures, simulating realistic noise levels (e.g., SNR 0.005) for controlled experiments. |
| **Preprocessing** | **(Prepares images for efficient model training)** |
| Patch-based approach | Divides large micrographs into smaller patches (e.g., $512 \times 512$ pixels). During training, random patches are used to save memory; during inference, overlapping patches are processed and merged with a soft weighting scheme for a seamless result. |
| **Modular Segmentation Pipeline** | **(The core deep learning framework for segmenting particles)** |
| Segmentation Models & Encoders | A flexible framework allowing users to combine various models and encoders. The default configuration uses a UNet++ model with an EfficientNet-B5 encoder.  [Supported segmentaion models](https://smp.readthedocs.io/en/latest/models.html) and [Supported encoders](https://smp.readthedocs.io/en/latest/encoders.html) |
| Loss Functions | Supports multiple loss functions, with Dice Loss as the default to handle class imbalance. It also includes Tversky Loss to fine-tune the trade-off between false positives and false negatives. [Supported Loss functions](https://smp.readthedocs.io/en/latest/losses.html) |
| Optimizer & Scheduler | Employs the Adam optimizer with a One-Cycle Learning rate scheduler to facilitate faster and more stable model training.  [Supported Optimizer](https://docs.pytorch.org/docs/stable/optim.html#algorithms) |
| **Spatial Regularization** | **(Refines the model's raw output for cleaner, more accurate boundaries)** |
| CRF / CD-CRF Layer | A Conditional Random Field (CRF) layer is used as a post-processing step to refine coarse segmentation maps by enforcing spatial consistency. The novel Class-Discriminative CRF (CD-CRF) uses learned, class-discriminative features from the CNN instead of raw pixel intensities, making it more robust to high noise levels in cryo-EM data. |
| CRF Solvers | Supports both the classic mean-field solver and a regularized Frank-Wolfe algorithm, which is the default due to its faster convergence and improved performance. |
| **Center-Finding** | **(Extracts final particle coordinates from the segmentation map)** |
| Traditional Computer Vision Methods | An algorithm that processes the segmentation map with normalization and erosion, detects object contours, filters them by area, and identifies particle centers from the minimal enclosing circle of each valid contour. |
| Crocker-Grier Algorithm | A classic blob detection method that identifies initial centers as local brightness maxima and refines their positions by calculating the intensity-weighted centroid. Overlapping blobs are resolved by discarding the one with the smaller "mass". |
| Non-Maximum Suppression (NMS) | A method where each pixel's probability is a confidence score. The algorithm retains only the highest-scoring candidate in local grids and then suppresses (discards) candidates that have significant overlap with a higher-scoring neighbor. |

---

## Tutorials and Guides

For detailed documentation and analysis on both synthetic and real datasets, check out our [Example Notebook](notebook/).

---

## Data

- **Synthetic Data**: Generate synthetic data using the scripts in the [simulation](simulation/) directory.
- **Tested Data**: Download tested data from [CryoPPP](https://github.com/BioinfoMachineLearning/cryoppp).

---

## License

CRISP is open-source software released under the [GNU General Public License, Version 3](https://github.com/phonchi/CryoParticleSegment/blob/master/LICENSE).

---

## Credits

- Builds upon the work developed by [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
- Integrates code from the [CRF - Conditional Random Fields](https://github.com/netw0rkf10w/CRF) project.
- The NMS and Morphology and contour finding algorithms are implemented according to the descriptions and built upon the description and code of [PIXER](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2614-y) and [CASSPER](https://www.nature.com/articles/s42003-021-01721-1).
