import os
from datetime import datetime

# ---------------------------
# ⚙️ 訓練參數
# ---------------------------
batch_size = 64
num_epochs = 50
learning_rate = 1e-3
num_workers = 0
shuffle = True

# ---------------------------
# 📐 模型架構參數
# ---------------------------
num_classes = 3
# num_classes = 4
input_subcarriers = 2025
pca_output_dim = 256  # 如果有用 PCA，這是降維後的維度
num_csi_channels = 1
tcn_channels = [128, 256, 512]
kernel_size = 5
dropout = 0.2

# ---------------------------
# 💾 儲存與輸出路徑
# ---------------------------
# 模型 / PCA 檔名帶入 timestamp
save_model_path = f"./checkpoints/env1_best_model_cnn.pth"