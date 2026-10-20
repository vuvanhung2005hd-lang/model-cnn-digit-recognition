"""
===============================================================================
File        : model.py
Project     : Handwritten Digit Recognition using CNN
===============================================================================
ARCHITECTURE DIAGRAM
===============================================================================
[INPUT]      Images size: 1 x 28 x 28 (Grayscale)
    │
[BLOCK 1]    Conv2d(kernel=5, stride=1)  6 x 24 x 24
    │        BatchNorm2d                 6 x 24 x 24
    │        ReLU                        6 x 24 x 24
    │        MaxPool2d(kernel=2)         6 x 12 x 12
    │
[BLOCK 2]    Conv2d(kernel=5, stride=1)  16 x 8 x 8
    │        BatchNorm2d                 16 x 8 x 8
    │        ReLU                        16 x 8 x 8
    │        MaxPool2d(kernel=2)         16 x 4 x 4
    │
[FLATTEN]    Flatten Tensor              256 (16 * 4 * 4)
    │
[FC 1]       Linear(256 ─> 120)          120
    │        ReLU                        120
    │
[FC 2]       Linear(120 ─> 84)           84
    │        ReLU                        84
    │
[DROPOUT]    Dropout(p)             
    │
[OUTPUT]     Linear(84 ─> 10)            10 (Logits)
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # =================================================
        # 1.Convolution Block 1
        # =================================================
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # =================================================
        # 2. Convolution Block 2
        # =================================================
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # =================================================
        # 3. Fully Connected) & Dropout
        # =================================================

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=16 * 4 * 4,
            out_features=120
        )
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(
            in_features=120,
            out_features=84
        )
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(
            p=0.5
        )

        self.fc3 = nn.Linear(
            in_features=84,
            out_features=10
        )

        # (Weight Initialization)
        self.initialize_weights()

    # =====================================================
    # Forward Propagation
    # =====================================================
    def forward(self, x):


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)


        x = self.flatten(x)


        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.relu4(x)


        x = self.dropout(x)


        x = self.fc3(x)

        return x

    # =====================================================
    #Weight Initialization
    # =====================================================
    def initialize_weights(self):
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# =========================================================
# Test Execution
# =========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MODERNIZED LENET-5 SUMMARY")
    print("=" * 70)

    model = CNN()
    print(model)
    print("=" * 70)

    #  MNIST/EMNIST
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)

    print(f"Input Shape  : {dummy_input.shape}")
    print(f"Output Shape : {output.shape} (10 predict class")
    print("=" * 70)
    print("MODEL TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)