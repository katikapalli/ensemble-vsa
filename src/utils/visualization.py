import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.utils.logger import TrainingLogger

plt.style.use('seaborn-v0_8')

def plot_loss_accuracy(model_name, log_dir="logs"):
    log_path = os.path.join(log_dir, f"{model_name}.json")

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    logger = TrainingLogger(model_name, log_dir)
    logs = logger.get_logs()
    
    if not logs:
        print(f"No logs available for {model_name}")
        return

    epochs = [entry["epoch"] for entry in logs]
    train_loss = [entry["train_loss"] for entry in logs]
    val_loss = [entry["val_loss"] for entry in logs]
    train_acc = [entry["train_acc"] for entry in logs]
    val_acc = [entry["val_acc"] for entry in logs]

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_loss, label="Train Loss", color='C1')
    plt.plot(epochs, val_loss, label="Validation Loss", color='C2')
    plt.title(f"Loss Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_acc, label="Train Accuracy", color='C1')
    plt.plot(epochs, val_acc, label="Validation Accuracy", color='C2')
    plt.title(f"Accuracy Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_labels, model_name="Model"):
    con_matrix = confusion_matrix(y_true, y_pred)
    counts = ["{0:0.0f}".format(value) for value in con_matrix.flatten()]
    percentages = ["{0:.1%}".format(value) for value in con_matrix.flatten() / np.sum(con_matrix, axis=1, keepdims=True).flatten()]
    labels = [f"{percentage}\n{count}" for count, percentage in zip(counts, percentages)]
    labels = np.asarray(labels).reshape(len(class_labels), len(class_labels))

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(con_matrix, annot=labels, fmt='', cmap='OrRd', square=True, xticklabels=class_labels, yticklabels=class_labels)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, class_labels, model_name="Model"):
    num_classes = len(class_labels)
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"], roc_auc["macro"] = all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})', linestyle=':', linewidth=4, color='deeppink')
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.2f})', linestyle=':', linewidth=4, color='navy')

    colors = ['violet', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'yellow']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.show()