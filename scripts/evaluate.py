import os
import sys
import yaml
import json
import argparse
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import AffectNetDataset
from src.models.dcnn import ProposedDCNN
from src.models.pretrained.densenet import DenseNet201
from src.models.pretrained.inception import InceptionV3
from src.models.pretrained.resnet import ResNet50
from src.models.pretrained.vgg16 import VGG16
from src.models.pretrained.vgg19 import VGG19
from src.models.pretrained.xception import Xception

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model(model_name, num_classes):
    models = {
        "ProposedDCNN": ProposedDCNN(num_classes=num_classes),
        "DenseNet201": DenseNet201(num_classes=num_classes),
        "InceptionV3": InceptionV3(num_classes=num_classes),
        "ResNet50": ResNet50(num_classes=num_classes),
        "VGG16": VGG16(num_classes=num_classes),
        "VGG19": VGG19(num_classes=num_classes),
        "Xception": Xception(num_classes=num_classes),
    }
    return models.get(model_name, ProposedDCNN(num_classes=num_classes))
    
def load_model(model_path, model_name, device, config):
    model = get_model(model_name, num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate(model, dataloader, device):
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    return accuracy, precision, recall, f1

def main(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = AffectNetDataset(root_dir=config['dataset']['dataset_root'], transform=transform, config=config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    model = load_model(config['callbacks']['checkpoint']['model_path'], config['model']['model_name'], device, config)
    accuracy, precision, recall, f1 = evaluate(model, test_loader, device)
    
    print("Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    main(args.config)
