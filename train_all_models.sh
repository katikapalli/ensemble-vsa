#!/bin/bash

CONFIG_FILES=(
    "configs/dcnn.yaml"
    "configs/inception.yaml"
    "configs/resnet.yaml"
    "configs/vgg16.yaml"
    "configs/vgg19.yaml"
    "configs/xception.yaml"
    "configs/densenet.yaml"
)

for config in "${CONFIG_FILES[@]}"; do
    echo "Training model with config: $config"
    python scripts/train.py --config "$config"
    echo "Finished training for: $config"
    echo "--------------------------------------"

done

echo "All models have been trained successfully!"
