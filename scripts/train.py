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
    
def setup_logger(model_name):
    """Sets up a logger with a separate log file for each model."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_name}.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
    
def save_model_performance(json_file, model_name, train_acc):
    data = {}
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    data[model_name] = {"train_accuracy": train_acc}
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

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

def get_loss_function(loss_name):
    loss_functions = {
        "CrossEntropyLoss": nn.CrossEntropyLoss(),
        "MSELoss": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
    }
    return loss_functions.get(loss_name, nn.CrossEntropyLoss())

def get_optimizer(optimizer_name, model, lr):
    optimizers = {
        "SGD": optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "Adam": optim.Adam(model.parameters(), lr=lr),
        "AdamW": optim.AdamW(model.parameters(), lr=lr),
    }
    return optimizers.get(optimizer_name, optim.Adam(model.parameters(), lr=lr))

def get_scheduler(optimizer, config):
    scheduler_name = config["callbacks"]["lr_scheduler"]["scheduler_name"]
    monitor = config["callbacks"]["lr_scheduler"]["monitor"]
    factor = config["callbacks"]["lr_scheduler"]["factor"]
    patience = config["callbacks"]["lr_scheduler"]["patience"]
    min_lr = config["callbacks"]["lr_scheduler"]["min_lr"]

    if scheduler_name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max" if "accuracy" in monitor else "min",
                                                    factor=factor, patience=patience, min_lr=min_lr, verbose=True)
    return None

def train(config_path):
    config = load_config(config_path)
    json_file = config["training"].get("model_performance_file", "model_performance.json")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = AffectNetDataset(root_dir=config['dataset']['dataset_root'], transform=transform, config=config, split='train')
    val_dataset = AffectNetDataset(root_dir=config['dataset']['dataset_root'], transform=transform, config=config, split='valid')

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config['model']['model_name']
    logger = setup_logger(model_name)
    logger.info(f"Starting training for model: {model_name}")
    model = get_model(config['model']['model_name'], config['dataset']['num_classes']).to(device)
    criterion = get_loss_function(config['training']['loss_fn'])
    optimizer = get_optimizer(config['training']['optimizer'], model, lr=config['optimizer']['lr'])
    scheduler = get_scheduler(optimizer, config)
    best_score = float('-inf') if config["callbacks"]["checkpoint"]["mode"] == "max" else float('inf')
    checkpoint_path = config["callbacks"]["checkpoint"]["model_path"]
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100 * correct_train / total_train
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = 100 * correct_val / total_val
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        monitor = config["callbacks"]["checkpoint"]["monitor"]
        mode = config["callbacks"]["checkpoint"]["mode"]
        current_score = val_accuracy if monitor == "val_accuracy" else val_loss

        if (mode == "max" and current_score > best_score) or (mode == "min" and current_score < best_score):
            print(f"Saving best model at epoch {epoch+1} with {monitor}: {current_score:.4f}")
            best_score = current_score
            torch.save(model.state_dict(), checkpoint_path)
            save_model_performance(json_file, model_name, train_accuracy)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if scheduler:
            scheduler.step(current_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deep learning model dynamically")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    train(args.config)
