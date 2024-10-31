# SpecVQGAN Fine-Tuning Experiments
## Overview
This repository contains the code used for fine-tuning the [SpecVQGAN](https://github.com/v-iashin/SpecVQGAN) model for coursework at the *National University of Kyiv-Mohyla Academy* titled *Exploration of Multimodal approaches in Image-to-Audio synthesis*. Our primary goal was to examine several deep learning observations within an image-to-audio domain through a comparative analysis of parameter configurations. Full detail on our coursework can be found here.

## Table of Contents
1. [Installation](#installation)
2. [Resources and Dataset](#resources-and-dataset)
3. [Evaluation](#evaluation)
4. [Results](#results)
5. [Credits](#credits)
6. [License](#license)


## Installation
This repository supports both Linux and Windows setups using `conda` virtual environments. We used PyTorch 2.2 and CUDA 12.1 for GPU-accelerated training. The setup files include configurations for Linux and Windows, as well as an optional Docker environment.

To set up the environment:
1. Choose the appropriate `conda` configuration file:
   - For Linux: `conda_env.yaml`
   - For Windows: `conda_env_win.yaml`

> [!NOTE]
> `conda_env` files use PyTorch 2.2 with CUDA version 12.1. If you're using a different version, or if you will be using ROCm or CPU for training, you might need to install them manually or modify the environment file according to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

2. Install the environment with the following command:
   ```bash
   conda env create -f conda_env.yaml  # For Linux
   conda env create -f conda_env_win.yaml  # For Windows
   ```
3. (Optional) A Dockerfile is provided for creating a Docker environment. The configuration may require updates, since it wasn't used in our experiments and was simply copied from the original repository.

## Resources and Dataset
We used the Visually Aligned Sounds (VAS) dataset for all of our experiments due to its small size, which allowed for quicker training and testing. Training was initially done on a personal RTX 3060 (6 GB VRAM) GPU but later scaled to a desktop with an RTX 3080 (12 GB VRAM) and finally a SLURM cluster using A100 and V100 GPUs with 40 and 80 GB VRAM.

## Evaluation
We used the same metrics used by the authors of SpecVQGAN to evaluate the fidelity and relevance: **Fréchet Inception Distance (FID)** and Melception-based **KL-divergence (MKL)** (lower is better).


## Results
The results and detailed explanations of our findings are available in the main Github README for the coursework. You can access it [here](https://github.com/Exploration-of-image-to-audio-synthesis/coursework-readme).


## Credits
This repository was forked from the [official SpecVQGAN repository](https://github.com/v-iashin/SpecVQGAN). Original paper, full documentation and usage examples can be found [here](https://github.com/v-iashin/SpecVQGAN).

## License
This project is licensed under the MIT License.