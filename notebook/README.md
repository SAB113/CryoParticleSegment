## CRISP Tutorial Series

Explore our comprehensive tutorial series designed to guide you through every stage of using the CRISP framework—from segmentation to downstream particle analysis. The series is divided into two main sections: the core segmentation pipeline and a practical downstream analysis example focused on particle picking.

### The dataset
> Please refer to the [table here](https://github.com/BioinfoMachineLearning/cryoppp?tab=readme-ov-file#cryoppp-statistics) for the available datasets in CryoPPP and their statistics.  
> For additional details about micrograph features in CryoPPP, see the [table in the paper](https://static-content.springer.com/esm/art%3A10.1038%2Fs41597-023-02280-2/MediaObjects/41597_2023_2280_MOESM4_ESM.xlsx).

### Main Segmentation Framework

- **[00-A_Mask_Generator_clean.ipynb](00-A_Mask_Generator_clean.ipynb)**
  - **Focus**: Generate segmentation maps for real datasets.
  - **Key Components**: Environment setup, dataset download, exploratory data analysis, application of various thresholding methods, and segmentation mask generation.
  - To proceed, please obtain a LICENSE ID of CryoSPARC from [this page](https://cryosparc.com/download).

- **[00-B_Dataset preprocessing.ipynb](00-B_Dataset%20preprocessing.ipynb)**
  - **Focus**: Create training, validation, and testing splits.
  - **Key Components**: Adjusting data ratios and storing datasets in NumPy format for streamlined processing.

- **[01_training_models_clean.ipynb](01_training_models_clean.ipynb)**
  - **Focus**: Train segmentation models using various architectures.
  - **Key Components**: Experimentation with different architectures, encoders, loss functions, and metrics, alongside performance visualization and benchmarking.
  - The platform is built on top of `segmentation_models.pytorch`. Most segmentation models, encoders, and loss functions are expected to work out of the box, although some untested options may require minor modifications. You can find the available [models here](https://smp.readthedocs.io/en/latest/models.html), [encoders here](https://smp.readthedocs.io/en/latest/models.html), and [loss functions here](https://smp.readthedocs.io/en/latest/losses.html). You can also customize your own modules, as described [here](https://smp.readthedocs.io/en/latest/insights.html).

- **[02_finetune_with_crf_clean.ipynb](02_finetune_with_crf_clean.ipynb)**
  - **Focus**: Fine-tune segmentation models with a CRF layer.
  - **Key Components**: Evaluating different CRF layers and solvers and performing joint training to refine segmentation results.

### Downstream Analysis – Particle Picking Example

- **[03_select_hyperparam_for_extraction_clean.ipynb](03_select_hyperparam_for_extraction_clean.ipynb)**
  - **Focus**: Optimize hyperparameters and center-finding algorithms.
  - **Key Components**: Testing three different center-finding methods with visualization to determine the best approach.

- **[04_particle_extraction_clean.ipynb](04_particle_extraction_clean.ipynb)**
  - **Focus**: Extract particles using the segmentation map and selected center-finding algorithm.
  - **Key Components**: Processing and converting extracted particles into the STAR format compatible with popular 3D reconstruction software.

- **[05_cryosparc.ipynb](05_cryosparc.ipynb)**
  - **Focus**: Perform 3D reconstruction.
  - **Key Components**: Setting up the environment and providing detailed instructions for executing 3D reconstruction workflows.
  - To proceed, please obtain a LICENSE ID of CryoSPARC from [this page](https://cryosparc.com/download).

