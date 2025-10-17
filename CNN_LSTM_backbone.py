import torch
import torch.nn as nn

class CNNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.relu(self.net(x) + self.skip(x))


class CNNBlock(nn.Module):
    """
    Matches the spirit of TCN_Block:
    - num_inputs: input channels
    - num_channels: list of out channels per layer, e.g., [64, 64, 128]
    - kernel_size, dropout as usual
    Keeps time length L unchanged.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            layers.append(CNNResidualBlock(in_ch, out_ch, kernel_size, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, C_in, L]
        return self.network(x)  # [B, C_last, L]

class LSTMBlock(nn.Module):
    """
    Stacked LSTM feature extractor with per-layer channel targets.
    - num_inputs: input channels at each time step
    - num_channels: list of desired output channels per layer; when bidirectional=True,
      each out_ch must be even (since out_ch = hidden_size * 2).
    - bidirectional: True gives stronger context (like BiTCN); False for causal-ish.
    Returns [B, C_last, L] to match your TCN/CNN interfaces.
    """
    def __init__(self, num_inputs, num_channels, dropout=0.2, bidirectional=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bidirectional = bidirectional

        dir = 2 if bidirectional else 1
        in_feat = num_inputs
        for i, out_ch in enumerate(num_channels):
            assert (out_ch % dir) == 0, \
                f"out_channels={out_ch} must be divisible by {dir} when bidirectional={bidirectional}"
            hidden = out_ch // dir
            self.layers.append(
                nn.LSTM(
                    input_size=in_feat,
                    hidden_size=hidden,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,            # dropout between stacked LSTMs handled below
                    bidirectional=bidirectional
                )
            )
            in_feat = out_ch  # the next layer receives out_ch (= hidden*dir)

        # Dropout applied on the temporal features between layers
        self.inter_drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C_in, L]
        B, C, L = x.shape
        x = x.permute(0, 2, 1)  # [B, L, C_in]
        for i, lstm in enumerate(self.layers):
            x, _ = lstm(x)       # [B, L, out_ch]
            if i < len(self.layers) - 1:
                x = self.inter_drop(x)
        x = x.permute(0, 2, 1).contiguous()  # [B, C_last, L]
        return x