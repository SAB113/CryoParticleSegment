# CRISP

**A Modular Framework for Cryo-EM Image Segmentation and Processing with Conditional Random Field**

CRISP is a modular framework that facilitates experimentation with advanced image segmentation strategies for cryo-electron microscopy (cryo-EM). It streamlines the process of generating high-quality segmentation maps and integrates seamlessly with downstream workflows.

---

## Features

- **Automated Label Generation**: Automatically create high-quality reference segmentation maps.
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
