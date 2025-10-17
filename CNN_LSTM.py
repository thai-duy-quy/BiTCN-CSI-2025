import torch
import torch.nn as nn
import torch.nn.functional as F
#from TCN_backbone import BidirectionalTCN, TCN_Block
from CNN_LSTM_backbone import CNNBlock, LSTMBlock
import numpy as np 

# num_inputs, num_channels, kernel_size=3, dropout=0.2
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, channels, kernel_size, dropout, num_classes, num_csi_channels=4):
        super().__init__()
        self.num_csi_channels = num_csi_channels
        #print(input_dim)
        self.cnn_per_channel = nn.ModuleList([
            CNNBlock(
                num_inputs=input_dim,
                num_channels=channels,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(num_csi_channels)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1] * num_csi_channels, num_classes)

    def forward(self, x):
        # x: [B, C, T, D]
        #print(x.shape)
        outputs = []
        for i in range(self.num_csi_channels):
            xi = x[:, i]               # [B, T, D]
            xi = xi.permute(0, 2, 1)   # [B, D, T]
            yi = self.cnn_per_channel[i](xi) 
            yi = self.pool(yi).squeeze(-1)
            outputs.append(yi)
        features = torch.cat(outputs, dim=1)
        logits = self.classifier(features)
        return logits

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, channels, dropout, num_classes, num_csi_channels=4):
        super().__init__()
        self.num_csi_channels = num_csi_channels
        #print(input_dim)
        self.cnn_per_channel = nn.ModuleList([
            LSTMBlock(
                num_inputs=input_dim,
                num_channels=channels,
                dropout=dropout,
                bidirectional = True
            ) for _ in range(num_csi_channels)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1] * num_csi_channels, num_classes)

    def forward(self, x):
        # x: [B, C, T, D]
        #print(x.shape)
        outputs = []
        for i in range(self.num_csi_channels):
            xi = x[:, i]               # [B, T, D]
            xi = xi.permute(0, 2, 1)   # [B, D, T]
            yi = self.cnn_per_channel[i](xi) 
            yi = self.pool(yi).squeeze(-1)
            outputs.append(yi)
        features = torch.cat(outputs, dim=1)
        logits = self.classifier(features)
        return logits