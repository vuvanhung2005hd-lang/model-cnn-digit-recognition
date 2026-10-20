"""
===============================================================================
File        : train.py
Project     : Handwritten Digit Recognition using CNN
Description :
    Training script for the modernized LeNet-5 model.
    Utilizes Adam optimizer, CrossEntropyLoss, and implements
    a best-model checkpointing mechanism.

===============================================================================
TRAINING PROCESS DIAGRAM
===============================================================================
[DATASET] --- (Images, Labels)
    │
    ├──> 1. FORWARD PASS  : outputs = model(images)
    │
    │
    ├──> 2. LOSS COMPUTE  : loss = criterion(outputs, labels)
    │
    │
    ├──> 3. BACKWARD PASS : loss.backward()
    │
    │
    └──> 4. OPTIMIZE STEP : optimizer.step()

===============================================================================
"""

# =========================================================
# Imports
# =========================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Import from custom modules
from model import CNN
from dataset import get_dataloader
from utils import plot_history

# =========================================================
# 1. Hyperparameters Configuration
# =========================================================
EPOCHS = 20                     # Number of training loops
LEARNING_RATE = 0.001             # Step size for optimizer

# Automatically select GPU if available, else use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory and path to save the best model weights
SAVE_DIR = "weights"
SAVE_PATH = os.path.join(SAVE_DIR, "best_model.pth")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# =========================================================
# 2. Main Training Function
# =========================================================
def train_model():
    print("=" * 70)
    print(" STARTING TRAINING PROCESS ")
    print("=" * 70)
    print(f"Device               : {DEVICE}")
    print(f"Number of Epochs     : {EPOCHS}")
    print(f"Learning Rate        : {LEARNING_RATE}")
    print("-" * 70)

    # Load DataLoaders
    train_loader, test_loader = get_dataloader()

    # Initialize Model and move to target device (GPU/CPU)
    model = CNN().to(DEVICE)

    # =========================================================
    # CORE MACHINE LEARNING COMPONENTS
    # =========================================================

    # 1. Loss Function
    criterion = nn.CrossEntropyLoss()

    # 2. Optimizer with Weight Decay
    # [VN] Thêm weight_decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 3. Learning Rate Scheduler

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # =========================================================

    # Track metrics history for plotting
    history_train_loss = []
    history_train_acc = []
    history_test_acc = []
    best_test_acc = 0.0

    # =========================================================
    # TRAINING LOOP
    # =========================================================
    for epoch in range(EPOCHS):

        # --- PHASE 1: TRAINING ---
        model.train()  # Enable Dropout & BatchNorm

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Step 1: Clear previous gradients
            optimizer.zero_grad()

            # Step 2: Forward pass
            outputs = model(images)

            # Step 3: Calculate loss
            loss = criterion(outputs, labels)

            # Step 4: Backward pass (compute gradients)
            loss.backward()

            # Step 5: Update weights
            #  Optimizer Adam
            optimizer.step()

            # Calculate training metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate epoch metrics and append to history lists
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        history_train_loss.append(epoch_loss)
        history_train_acc.append(train_acc)

        # --- PHASE 2: EVALUATION ---
        model.eval()   # Disable Dropout 

        correct_test = 0
        total_test = 0

        # Disable gradient calculation for faster inference
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test
        history_test_acc.append(test_acc)

        # Print epoch summary
        print(f"Epoch [{epoch + 1:2d}/{EPOCHS}] | "
              f"Loss: {epoch_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        # --- PHASE 3: BEST CHECKPOINT SAVING ---

        scheduler.step()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" --> Best model saved! (Test Acc: {best_test_acc:.2f}%) to '{SAVE_PATH}'")

    print("=" * 70)
    print(" TRAINING COMPLETED!")
    print(f" Highest Test Accuracy: {best_test_acc:.2f}%")
    print("=" * 70)

    # Plot and save the performance curves using utils.py
    plot_history(history_train_loss, history_train_acc, history_test_acc, SAVE_DIR)


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    train_model()