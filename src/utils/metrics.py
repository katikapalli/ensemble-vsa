import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_metrics(y_true, y_pred, y_pred_proba, num_classes):
    results = {}

    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision_weighted"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    results["recall_weighted"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    results["f1_weighted"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    results["precision_per_class"] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    results["recall_per_class"] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    results["f1_per_class"] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    try:
        results["roc_auc_weighted"] = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')
        results["roc_auc_per_class"] = roc_auc_score(y_true, y_pred_proba, average=None, multi_class='ovr').tolist()
    except ValueError:
        results["roc_auc_weighted"] = None
        results["roc_auc_per_class"] = None

    return results

def evaluate_model(model, dataloader, criterion, device, num_classes):
    model.eval()
    y_true, y_pred, y_pred_proba = [], [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            _, predicted = outputs.max(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / total_samples

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    metrics = compute_metrics(y_true, y_pred, y_pred_proba, num_classes)
    metrics["loss"] = avg_loss

    return metrics
