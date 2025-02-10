import os
import json
import sys
import yaml
import argparse
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

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

def load_models(model_dir, model_names):
    models = {}
    for model_name in model_names:
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        model = get_model(model_name, num_classes=7)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models[model_name] = model
    return models

def generate_predictions(models, dataloader, device):
    predictions = {}
    true_labels = []
    
    for model_name, model in models.items():
        model.to(device)
        model_predictions = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                outputs = model(images)
                model_predictions.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                if model_name == list(models.keys())[0]:
                    true_labels.extend(labels.cpu().numpy())
        
        predictions[model_name] = np.array(model_predictions)
    
    return predictions, np.array(true_labels)

def majority_voting(predictions, selected_models):
    num_samples = predictions[selected_models[0]].shape[0]
    final_predictions = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        model_preds = [np.argmax(predictions[model][i]) for model in selected_models]
        unique, counts = np.unique(model_preds, return_counts=True)
        
        if len(unique) == len(selected_models) or max(counts) == 1:
            avg_softmax_scores = np.mean([predictions[model][i] for model in selected_models], axis=0)
            final_predictions[i] = np.argmax(avg_softmax_scores)
        else:
            final_predictions[i] = unique[np.argmax(counts)]
    
    return final_predictions

def evaluate_ensemble(y_true, y_pred, args):
    print(classification_report(y_true, y_pred))
    print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred, average="weighted"):.4f}')
    print(f'Recall: {recall_score(y_true, y_pred, average="weighted"):.4f}')
    print(f'F1-Score: {f1_score(y_true, y_pred, average="weighted"):.4f}')
    
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='OrRd', square=True, xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Average Voting')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')

    filename = f'outputs/majority_voting_{"proposed" if args.use_proposed_dcnn else "baseline"}_{args.num_models}.png'
    plt.savefig(filename)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--use_proposed_dcnn', type=bool, default=False, help='Whether to include the proposed DCNN model')
    parser.add_argument('--num_models', type=int, default=2, help='Number of top models to use in the ensemble')
    parser.add_argument('--model_performance', type=str, default='model_performance.json', help='Path to JSON file with model accuracies')
    args = parser.parse_args()
    
    config = load_config(args.config)
    model_dir = "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.model_performance, 'r') as f:
        model_performance = json.load(f)
    
    sorted_models = sorted(model_performance.items(), key=lambda x: x[1]['train_accuracy'], reverse=True)
    
    if args.use_proposed_dcnn:
        top_models = [model[0] for model in sorted_models[: args.num_models]]
    else:
        filtered_models = [model[0] for model in sorted_models if model[0] != "ProposedDCNN"]
        top_models = filtered_models[: args.num_models]
    
    models = load_models(model_dir, top_models)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valid_dataloader = AffectNetDataset(root_dir=config['dataset']['dataset_root'], transform=transform, config=config, split='valid')
    valid_dataloader = DataLoader(valid_dataloader, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

    predictions, y_true = generate_predictions(models, valid_dataloader, device)
    y_pred = majority_voting(predictions, top_models)
    
    print(f'Using models: {top_models}')
    evaluate_ensemble(y_true, y_pred, args)

if __name__ == "__main__":
    main()
