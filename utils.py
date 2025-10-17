# utils.py
import matplotlib.pyplot as plt
import matplotlib
import torch
# matplotlib.use("Agg")  # 強制使用無需 GUI 的後端
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from torch_geometric.utils import from_scipy_sparse_matrix
#from sklearn.neighbors import NearestNeighbors
#from scipy.sparse import csr_matrix

def plot_training_curve(train_losses, val_losses, train_accuracies, val_accs, save_path="training_plot.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, val_losses, label='Val loss')
    plt.plot(epochs, train_accuracies, label='Train accuracy')
    plt.plot(epochs, val_accs, label='Val accuracy')
    plt.xlabel('Epoch')
    plt.title('Loss and Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()                   # 一定要關閉

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    # 計算並正規化 confusion matrix（按列，也就是每個 true label）
    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100  # 百分比化

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",  # 顯示一位小數
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Accuracy (%)'}
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Percentage %)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")

def apply_gmm_mask(x, gmm_model):
    B, T, D = x.shape
    x_flat = x.reshape(-1, D).numpy()
    labels = gmm_model.predict(x_flat)
    mask = torch.tensor(labels.reshape(B, T), dtype=torch.float32).unsqueeze(-1)
    return x * mask  # [B, T, D]

def build_edge_index(num_actions):
    edge_list = []
    for i in range(num_actions):
        for j in range(num_actions):
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list).t().contiguous()  # [2, E]
    return edge_index