import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super(ResidualBlock, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        # Shortcut connection (identity mapping)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        # Shortcut connection
        shortcut = self.shortcut(x)

        # Add the residual and apply ReLU
        out += shortcut
        out = self.relu(out)

        return out


class TranscriptionCNN(nn.Module):
    def __init__(self, vocab_size, d_model, filter_count, num_residual_blocks, device, dropout_prob=0.0):
        super(TranscriptionCNN, self).__init__()

        self.conv1 = nn.Conv1d(d_model, vocab_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(filter_count)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.res_blocks = self._make_residual_blocks(filter_count, filter_count, num_residual_blocks, dropout_prob)
        self.out_conv = nn.Conv1d(filter_count, vocab_size, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.device = device

    def _make_residual_blocks(self, in_channels, out_channels, num_blocks, dropout_prob):
        blocks = []
        for i in range(num_blocks):
            stride = 1
            blocks.append(ResidualBlock(in_channels, out_channels, stride=stride, dropout_prob=dropout_prob))
            in_channels = out_channels  # Update in_channels for the next block
        return nn.Sequential(*blocks)

    def forward_new(self, x):
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        res_out = self.res_blocks(out)

        output = self.out_conv(res_out)
        output = self.dropout2(output)
        output = output.permute(0, 2, 1)

        return output

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        #out = self.relu(out)
        #out = self.dropout1(out)

        #output = self.out_conv(out)
        #output = self.dropout2(output)
        output = out.permute(0, 2, 1)

        return output
