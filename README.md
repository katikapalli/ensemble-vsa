# Enhancing Visual Sentiment Analysis with Ensemble-weighted Deep Convolutional Neural Networks

[![DOI](https://zenodo.org/badge/930352173.svg)](https://doi.org/10.5281/zenodo.14845110)

**Note:** This is a PyTorch implementation of the original paper. Some variations in results may occur due to implementation differences.

## Description

<hr />

> **Abstract:** *Visual Sentiment Analysis (VSA) is crucial for understanding human emotions conveyed through visual materials such as images and films. This study aims to develop a robust VSA system by integrating deep learning and ensemble techniques. We propose an enhanced deep convolutional neural network (DCNN) to automatically extract facial features from images without the need for manual feature engineering. To further boost performance, we introduce a novel ensemble learning approach using majority, average, and weighted voting mechanisms. This ensemble technique effectively reduces biases and variations of individual models, leading to improved accuracy and robustness. Extensive experiments on the AffectNet dataset demonstrate the efficacy of our proposed method. Our ensemble-weighted voting approach achieves an accuracy of 70%, outperforming current state-of-the-art techniques, while the standalone DCNN model attains an accuracy of 69.28%. This study not only advances the field of VSA but also highlights the potential of ensemble learning in enhancing deep learning models for complex visual tasks.* 

<hr />

## Network Architecture

![architecture](weightedVoting.png?raw=true)

## Setup

Before training the models, clone the repository, navigate to the directory, create a **Conda environment**, and install dependencies:

```sh
git clone https://github.com/katikapalli/ensemble-vsa.git
cd ensemble-vsa
```

```sh
conda create --name ensemble_vsa python=3.10.15 -y
conda activate ensemble_vsa
pip install -r requirements.txt
```

## Training

To train individual models or all models at once, use the provided shell script.

### Train Individual Model

To train a specific model, run:


```sh
python scripts/train.py --config configs/<model_config>.yaml
```

Replace `<model_config>` with the corresponding configuration file.

#### Example:
For training a **ResNet** model:

```sh
python scripts/train.py --config configs/resnet.yaml
```

### Train All Models

To train all models sequentially, run:

```sh
chmod +x train_all_models.sh
```

```sh
./train_all_models.sh
```

## Evaluation

After training, evaluate the model using:

```sh
python scripts/evaluate.py --config configs/<model_config>.yaml
```

## Ensemble Learning

Ensemble learning improves performance by combining multiple models.

### Majority Voting

Run majority voting with:

```sh
python src/ensemble/majority.py --config configs/config.yaml --use_proposed_dcnn True --num_models 5
```

Arguments:

- `--config` : Path to the config file.
- `--use_proposed_dcnn` : Use Proposed model or not
- `--num_models` : Number of models used in the ensemble.

### Average Voting

Run average voting with:

```sh
python src/ensemble/average.py --config configs/config.yaml --use_proposed_dcnn False --num_models 5
```

Arguments:

- `--config` : Path to the config file.
- `--use_proposed_dcnn` : Use Proposed model or not
- `--num_models` : Number of models used in the ensemble.

### Weighted Voting

Run weighted voting with:

```sh
python src/ensemble/weighted.py --config configs/config.yaml --use_proposed_dcnn True --num_models 5
```

Arguments:

- `--config` : Path to the config file.
- `--use_proposed_dcnn` : Use Proposed model or not
- `--num_models` : Number of models used in the ensemble.
- `--step` : Smallest difference possible between the weights; lower values result in more refined weight assignments.

Ensure all models are trained before running ensemble methods. Modify arguments as needed for customization.

## Citation

If our codebase or paper contributes to your research, please consider citing our paper.

```bibtex
@article{lokesh2025enhancing,
  author    = {K. Lokesh and Gaurav Meena},
  title     = {Enhancing Visual Sentiment Analysis with Ensemble-weighted Deep Convolutional Neural Networks},
  journal   = {The Visual Computer},
  publisher = {Springer Nature},
  year      = {forthcoming},
  note      = {In press}
}

@software{katikapalli_lokesh_2025_14845122,
  author       = {Katikapalli Lokesh},
  title        = {katikapalli/ensemble-vsa: Pre-Release: Ensemble-weighted Deep CNN (v1.0.0-beta)},
  year         = {2025},
  version      = {v1.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14845122},
  url          = {https://doi.org/10.5281/zenodo.14845122}
}

```

## Our Related Work

**FSTL-SA: Few-Shot Transfer Learning for Sentiment Analysis from Facial Expressions** [Paper](https://link.springer.com/article/10.1007/s11042-024-20518-y)  
Published in *Multimedia Tools and Applications*, 2024.  


## Contact

For any inquiries or further discussions, feel free to contact at [katikapalli.lokesh@yahoo.com](mailto:katikapalli.lokesh@yahoo.com). Cheers!
