# Enhancing Visual Sentiment Analysis with Ensemble-weighted Deep Convolutional Neural Networks

[![DOI](https://zenodo.org/badge/930352173.svg)](https://doi.org/10.5281/zenodo.14845110)

## Overview
This repository provides a **PyTorch implementation** of the paper *Enhancing Visual Sentiment Analysis with Ensemble-weighted Deep Convolutional Neural Networks*. Some variations in results may occur due to implementation differences.


## Dataset

The AffectNet dataset is not included in this repository. You can obtain the dataset from the **official AffectNet website**.

- AffectNet Dataset: https://mohammadmahoor.com/affectnet/
- Original Publication: [AffectNet Paper](https://ieeexplore.ieee.org/document/8013713)

## Network Architecture

Our deep learning pipeline consists of multiple models trained individually and then combined using an ensemble approach.

![architecture](weightedVoting.png?raw=true)

## Installation Guide


Before training the models, clone the repository, navigate to the directory, create a **Conda environment**, and install dependencies:

### Clone the Repository

```sh
git clone https://github.com/katikapalli/ensemble-vsa.git
cd ensemble-vsa
```

### Setup Conda Environment and Install Dependencies

```sh
conda create --name ensemble_vsa python=3.10.15 -y
conda activate ensemble_vsa
pip install -r requirements.txt
```

## Model Training

We train multiple models independently and then aggregate predictions using ensemble techniques.


### Trained Models

The following models are included:

| Model         | Configuration File           |
|--------------|------------------------------|
| ResNet       | `configs/resnet.yaml`        |
| VGG16        | `configs/vgg16.yaml`         |
| VGG19        | `configs/vgg19.yaml`         |
| Xception     | `configs/xception.yaml`         |
| EfficientNet | `configs/efficientnet.yaml`  |
| DenseNet     | `configs/densenet.yaml`      |
| Proposed DCNN| `configs/proposed_dcnn.yaml` |


### Train Individual Model


```sh
python scripts/train.py --config configs/<model_config>.yaml
```

Replace `<model_config>` with the corresponding configuration file.

#### Example:
For training a **ResNet** model:

```sh
python scripts/train.py --config configs/resnet.yaml
```

### Train All Models Sequentially 

```sh
chmod +x train_all_models.sh
```

```sh
./train_all_models.sh
```

## Model Evaluation

Once training is complete, evaluate a model using:

```sh
python scripts/evaluate.py --config configs/<model_config>.yaml
```

## Ensemble Learning

We employ multiple ensemble strategies to improve performance.

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
python src/ensemble/weighted.py --config configs/config.yaml --use_proposed_dcnn True --num_models 5 --step 0.01
```

Arguments:

- `--config` : Path to the config file.
- `--use_proposed_dcnn` : Use Proposed model or not
- `--num_models` : Number of models used in the ensemble.
- `--step` : Smallest difference possible between the weights; lower values result in more refined weight assignments.

Ensure all models are trained before running ensemble methods. Modify arguments as needed for customization.

## Key Algorithms and Implementation

### 1. Deep Convolutional Neural Networks (DCNNs)

Each model extracts visual sentiment features from images using deep feature extraction techniques such as convolutional and pooling layers.

### 2. Ensemble Learning

We use three ensemble techniques to combine predictions from multiple models:

- **Majority Voting**: Selects the most frequent prediction among models.
- **Average Voting**: Computes the mean probability score across models.
- **Weighted Voting**: Assigns weights based on model confidence to improve final predictions.

### 3. Weighted Voting Strategy

Weighted voting assigns different importance levels to models:

- Higher weight for more accurate models
- Lower weight for less confident models
- Optimization step to adjust weight precision




## Citation

If our repository helps in your research, please cite our paper.

```bibtex

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
