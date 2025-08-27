# Micrograph Simulation

This module provides tools for generating synthetic cryo-EM micrographs with controlled noise levels and ground truth annotations. It enables researchers to create datasets for training and evaluating segmentation algorithms under known conditions.

## Required Packages:
- mrcfile
- starfile
- scikit-image
- aspire

## Overview:

The simulation pipeline consists of three main components:
1. **Clean micrograph generation**: Creates noise-free synthetic micrographs from 3D volumes
2. **Noise addition**: Adds realistic noise at specified Signal-to-Noise Ratio (SNR) levels
3. **Ground truth generation**: Creates perfect segmentation masks for training and evaluation

## Usage:

### Tutorials

Referring to [micrograph_simulation.ipynb](micrograph_simulation.ipynb) for the details about synthetic dataset generation.

### Command Line Tools

#### Generate Clean Simulated Micrographs

Generate noise-free synthetic micrographs from a 3D volume and refinement results:

```bash
python NoisyImageGenerator.py volume_path refine_result_path simulated_micrograph_directory
```

This command will generate simulated micrographs without noise in the specified directory.

Directory structure:

```bash
simulated_micrograph_directory/
├── sim_image_0.mrc
├── sim_image_0.star
├── sim_image_1.mrc
├── sim_image_1.star
│   ...
├── sim_image_83.mrc
└── sim_image_83.star
```

#### Split the Dataset

The generated simulated micrographs should be split into train-validation-test manually:

Directory structure:

```bash
simulated_micrograph_directory/
├── train/
│   ├── sim_image_0.mrc
│   ├── sim_image_0.star
│   │   ...
│   ├── sim_image_57.mrc
│   └── sim_image_57.star
├── val/
│   ├── sim_image_58.mrc
│   ├── sim_image_58.star
│   │   ...
│   ├── sim_image_66.mrc
│   └── sim_image_66.star
└── test/
    ├── sim_image_67.mrc
    ├── sim_image_67.star
    │   ...
    ├── sim_image_83.mrc
    └── sim_image_83.star
```

#### Add Realistic Noise

Add noise to clean synthetic micrographs at a specified Signal-to-Noise Ratio (default: 0.1):

```bash
python NoisyImageGenerator.py simulated_micrograph_directory noisy_image_directory -snr 0.1
```

This command will generate simulated micrographs with noise in the specified directory.

Directory structure:

```bash
noisy_image_directory/
├── train/
│   ├── sim_image_0.mrc
│   ├── sim_image_0.star
│   │   ...
│   ├── sim_image_57.mrc
│   └── sim_image_57.star
├── val/
│   ├── sim_image_58.mrc
│   ├── sim_image_58.star
│   │   ...
│   ├── sim_image_66.mrc
│   └── sim_image_66.star
└── test/
    ├── sim_image_67.mrc
    ├── sim_image_67.star
    │   ...
    ├── sim_image_83.mrc
    └── sim_image_83.star
```

#### Create Ground Truth Masks

Generate perfect segmentation masks for the synthetic micrographs:

```bash
python GroundTruthGenerator.py simulated_micrograph_directory ground_truth_directory
```

This generates binary masks indicating particle locations for training segmentation models.

**Output structure:**

```bash
ground_truth_directory/
├── sim_image_0.mrc
├── sim_image_0.star
│   ...
├── sim_image_83.mrc
└── sim_image_83.star
```

## Notes:

- All generated files maintain the `.mrc` format for images and `.star` format for particle coordinates
- The simulation pipeline is designed to work with the ASPIRE library for realistic cryo-EM physics
- Ground truth masks provide pixel-perfect annotations for supervised learning
