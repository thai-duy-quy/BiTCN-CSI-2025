import torch
import torch.nn as nn
import torch.nn.functional as F
from TCN_backbone import BidirectionalTCN, TCN_Block
import numpy as np 

class BiTCNClassifier(nn.Module):
    def __init__(self, input_dim, tcn_channels, kernel_size, dropout, num_classes, num_csi_channels=4):
        super().__init__()
        self.num_csi_channels = num_csi_channels
        #print(input_dim)
        self.tcn_per_channel = nn.ModuleList([
            BidirectionalTCN(
                num_inputs=input_dim,
                num_channels=tcn_channels,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(num_csi_channels)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(tcn_channels[-1] * num_csi_channels, num_classes)

    def forward(self, x):
        # x: [B, C, T, D]
        #print(x.shape)
        outputs = []
        for i in range(self.num_csi_channels):
            xi = x[:, i]               # [B, T, D]
            #print(xi.shape)
            xi = xi.permute(0, 2, 1)   # [B, D, T]
            #print(xi.shape)
            yi = self.tcn_per_channel[i](xi) 
            #print(yi.shape)
            yi = self.pool(yi).squeeze(-1)
            #print(yi.shape)
            #exit()
            outputs.append(yi)
        #outputs1 = np.array(outputs)
        #print(outputs.shape)
        #exit()
        features = torch.cat(outputs, dim=1)
        logits = self.classifier(features)
        return logits

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, tcn_channels, kernel_size, dropout, num_classes, num_csi_channels=1):
        super().__init__()
        self.num_csi_channels = num_csi_channels
        #print(input_dim)
        self.tcn_per_channel = nn.ModuleList([
            TCN_Block(
                num_inputs=input_dim,
                num_channels=tcn_channels,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(num_csi_channels)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(tcn_channels[-1] * num_csi_channels, num_classes)

    def forward(self, x):
        # x: [B, C, T, D]
        #print(x.shape)
        outputs = []
        for i in range(self.num_csi_channels):
            xi = x[:, i]               # [B, T, D]
            xi = xi.permute(0, 2, 1)   # [B, D, T]
            yi = self.tcn_per_channel[i](xi) 
            yi = self.pool(yi).squeeze(-1)
            outputs.append(yi)
        features = torch.cat(outputs, dim=1)
        logits = self.classifier(features)
        return logits