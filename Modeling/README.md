# CRISP Modeling Module

This directory contains the core modeling components of the CRISP framework for cryo-EM image segmentation. It provides a comprehensive suite of tools for training, evaluating, and deploying deep learning models with advanced post-processing capabilities.

## Module Overview

### Core Components

| File | Description |
|------|-------------|
| [`model.py`](model.py) | Model factory functions and wrapper classes for backbone networks with CRF integration |
| [`trainer.py`](trainer.py) | Training and evaluation classes including specialized CryoEM trainers with patch-based processing |
| [`dataset.py`](dataset.py) | Dataset classes and data loading utilities for cryo-EM micrographs |
| [`metrics.py`](metrics.py) | Evaluation metrics specific to segmentation tasks |
| [`lr_scheduler.py`](lr_scheduler.py) | Learning rate scheduling and callback mechanisms |
| [`center_finding.py`](center_finding.py) | Particle center detection algorithms for downstream analysis |
| [`plot.py`](plot.py) | Visualization utilities for results and training progress |
| [`utils.py`](utils.py) | General utility functions and helper methods |
| [`convcrf.py`](convcrf.py) | Convolutional CRF implementation for spatial regularization |

### Specialized Subdirectories

- **[`CRF_main/`](CRF_main/)**: Advanced Conditional Random Field implementations with Frank-Wolfe and mean-field solvers

## Key Features

### Model Architecture Support
- **Flexible Backbone Integration**: Works with any segmentation model from `segmentation_models_pytorch`
- **CRF Post-processing**: Advanced spatial regularization using Gaussian CRF and Frank-Wolfe solvers
- **Multi-class Segmentation**: Support for binary and multi-class particle segmentation

### Training Framework
- **Patch-based Processing**: Memory-efficient training on large cryo-EM micrographs
- **Advanced Schedulers**: One-cycle learning rate scheduling and early stopping
- **Comprehensive Metrics**: IoU, Dice score, precision, recall, and F1-score tracking
- **GPU Optimization**: Efficient GPU memory management for large-scale training

### Center Finding Algorithms
- **Traditional CV Methods**: Contour-based particle detection with morphological operations
- **Crocker-Grier Algorithm**: Classic blob detection with intensity-weighted centroids
- **Non-Maximum Suppression**: Confidence-based particle detection with overlap resolution

## Usage Examples

### Basic Model Creation
```python
import segmentation_models_pytorch as smp
from model import create_model, create_crf_model

# Create a UNet++ backbone
backbone = smp.UnetPlusPlus(encoder_name="efficientnet-b5", classes=2)

# Wrap with output standardization
model = create_model(backbone, addout=True)

# Add CRF post-processing
crf_model = create_crf_model(backbone, crf_config, shape=(512, 512), num_classes=2)
```

### Training Setup
```python
from trainer import CryoEMTrainer
import torch.optim as optim

trainer = CryoEMTrainer(
    model=model,
    train_dataset=train_data,
    criterion=criterion,
    optimizer=optim.Adam(model.parameters()),
    device='cuda',
    metrics=['loss', 'iou'],
    num_classes=2
)

# Train the model
trainer.train(num_epochs=100, batch_size=16)
```

### Particle Detection
```python
from center_finding import normalize, pad_image
import numpy as np

# Preprocess segmentation map
normalized_map = normalize(segmentation_output)
padded_map = pad_image(normalized_map)

# Apply center finding algorithm
particle_centers = find_particle_centers(padded_map)
```

## Dependencies

- PyTorch >= 1.6
- segmentation_models_pytorch
- OpenCV (cv2)
- NumPy
- scikit-image
- CUDA >= 10.1 (for GPU acceleration)

## Integration with CRISP Pipeline

This modeling module integrates seamlessly with:
- **Mask Generation Pipeline**: Uses generated masks for supervised training
- **Preprocessing Tools**: Handles patch-based data loading and augmentation
- **Center Finding**: Converts segmentation maps to particle coordinates
- **Evaluation Framework**: Provides comprehensive performance assessment

For complete usage examples, refer to the [notebook tutorials](../notebook/) that demonstrate end-to-end workflows from training to particle extraction.