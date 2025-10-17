import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from load_data_frame import get_dataloaders
from load_data_widar import get_dataloaders_widar
from TCN import BiTCNClassifier,TCNClassifier
from CNN_LSTM import CNNClassifier, LSTMClassifier
from utils import plot_training_curve, plot_confusion_matrix, build_edge_index
from hyperparameter import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train import train
from eval import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="TCN Training & Evaluation")
    parser.add_argument("--eval-only", action="store_true", default=False, help="只進行測試，不執行訓練")
    parser.add_argument("--load-model", type=str, default=None, help="載入已儲存模型的路徑")
    parser.add_argument("--early-stop", type=int, default=20, help="若驗證集準確率連續 n 次未提升則提前停止訓練")
    parser.add_argument("--eval-env", type=str, default="env1", help="要用哪個環境做測試 (env1 或 env2)")
    parser.add_argument("--eval-root", type=str, default="./env_frame_pca", help="對應 eval 環境的路徑")
    return parser.parse_args()


def load_data():
    return get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        train_split=0.75,
    )
def build_model(device):
    # model = LSTMClassifier(
    #     input_dim=400,                
    #     channels=tcn_channels,    
    #     dropout=dropout,
    #     num_classes=num_classes,
    #     num_csi_channels=1            
    # ).to(device)
    # model = CNNClassifier(
    #     input_dim=400,                # 每個通道的子載波數
    #     channels=tcn_channels,    # TCN 每層通道數 list
    #     kernel_size=kernel_size,
    #     dropout=dropout,
    #     num_classes=num_classes,
    #     num_csi_channels=1            # 對應你 reshape 出來的 4 個通道
    # ).to(device)
    # model = TCNClassifier(
    #     input_dim=400,                # 每個通道的子載波數
    #     tcn_channels=tcn_channels,    # TCN 每層通道數 list
    #     kernel_size=kernel_size,
    #     dropout=dropout,
    #     num_classes=num_classes,
    #     num_csi_channels=1            # 對應你 reshape 出來的 4 個通道
    # ).to(device)
    model = BiTCNClassifier(
        input_dim=400,                # 每個通道的子載波數
        tcn_channels=tcn_channels,    # TCN 每層通道數 list
        kernel_size=kernel_size,
        dropout=dropout,
        num_classes=num_classes,
        num_csi_channels=1            # 對應你 reshape 出來的 4 個通道
    ).to(device)
    return model

def save_model(env,model,data):
    save_model_path = f"./checkpoints/{model}_env{env}_best_model_{data}.pth"
    return save_model_path

env = '1'
md = 'CNN'
dataset_train = 'widar'
def train_model(model, dataloader, val_loader, criterion, optimizer, scheduler, device, early_stop_patience):
    best_acc = 0.0
    no_improve_epochs = 0
    train_losses,val_losses, train_accuracies, val_accs = [], [], [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        loss, acc = train(model, dataloader, criterion, optimizer, device, epoch=epoch + 1)
        train_losses.append(loss)
        train_accuracies.append(acc)

        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch+1}] Train Acc: {acc:.2f}%, Val Acc: {val_acc:.2f}%")
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            #no_improve_epochs = 0
            save_model_path = save_model(env,md,dataset_train)
            torch.save(model.state_dict(), save_model_path)
            print(f"Saved best model to {save_model_path}")
        # else:
        #     no_improve_epochs += 1
        #     if no_improve_epochs >= early_stop_patience:
        #         print("⏹Early stopping triggered.")
        #         break

    elapsed = time.time() - start_time
    print(f"Training completed in {int(elapsed // 60)} min {int(elapsed % 60)} sec")
    return train_losses, val_losses, train_accuracies, val_accs


def evaluate_model(model, val_loader, criterion, device, epoch=None):
    val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device, epoch=epoch)
    print(f"[Epoch {epoch}] Final Eval → Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    class_names = [f"A{i}" for i in range(1, num_classes + 1)]
    save_path=f"confusion_matrix_env{env}_{md}.png"
    plot_confusion_matrix(labels, preds, class_names,save_path=save_path)


def main():
    args = parse_args()
    print(args)
    # exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    # 訓練使用 env1
    # train_loader, val_loader = get_dataloaders(
    #     env_list=["env1"],
    #     env_roots={"env1": "./env1_frame_pca"},
    #     batch_size=8,
    #     csv_path="Action_Segment.csv"
    # )


    if args.eval_only:
        # 🔍 只做測試時載入 eval 環境資料
        val_loader = get_dataloaders(
            env_list=[args.eval_env],
            env_roots={args.eval_env: args.eval_root},
            batch_size=8,
            csv_path="Action_Segment.csv"
        )[0]  # 只用 val_loader 即可
    else:
        # 🧪 訓練使用 env1
        # train_loader, val_loader = get_dataloaders(
        #     env_list=["env1"],
        #     env_roots={"env1": "./env1_frame_pca"},
        #     batch_size=8,
        #     csv_path="Action_Segment.csv"
        # )
        train_loader, val_loader = get_dataloaders_widar()

    if args.eval_only:
        # 測試階段只需要載入模型與 loss
        assert args.load_model is not None, "❌ 必須指定 --load-model 才能使用 --eval-only 模式"
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded model from {args.load_model}")
        criterion = nn.CrossEntropyLoss()
        evaluate_model(model, val_loader, criterion, device, epoch="Test")
        return

    
    # criterion = nn.CrossEntropyLoss()
     # ✅ Step 1: 收集所有 training label
    train_labels = []
    for _, label in train_loader.dataset:
        train_labels.append(label)

    labels_in_y = np.unique(train_labels)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=labels_in_y,
        y=train_labels
    )

    # 建立 full 長度的 weight tensor
    full_weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls, weight in zip(labels_in_y, class_weights):
        full_weights[cls] = weight

    criterion = nn.CrossEntropyLoss(weight=full_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded model from {args.load_model}")

    if not args.eval_only:
        train_losses, val_losses, train_accuracies, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.early_stop)
        #print(accs)
        save_path = f'training_plot_env{env}_{md}.png'
        plot_training_curve(train_losses, val_losses, train_accuracies, val_accs,save_path=save_path)
        dataset_train = 'widar'
        save_model_path = save_model(env,md,data=dataset_train)
        model.load_state_dict(torch.load(save_model_path))
        print(f"Loaded best model from {save_model_path}")

    evaluate_model(model, val_loader, criterion, device, epoch=num_epochs)

if __name__ == "__main__":
    # m = nn.Conv1d(16,33,3,stride=2)
    # input = torch.randn(20,16,3)
    # output = m(input)
    # print(output)

    main()