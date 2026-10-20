"""
===============================================================================
File        : utils.py
Project     : Handwritten Digit Recognition using CNN
Description :
    Utility functions for plotting training metrics with clean integer X-axis.
===============================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  

def plot_history(train_loss, train_acc, test_acc, save_dir):
    print("\n[INFO] Generating and saving performance plots...")

    epochs = range(1, len(train_acc) + 1)

    # =========================================================
    # 1. Plot Accuracy Curve
    # =========================================================
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epochs, test_acc, label='Test Accuracy', color='orange', marker='s')

    plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    acc_path = os.path.join(save_dir, 'accuracy_curve.png')
    plt.savefig(acc_path)
    print(f"[INFO] Saved Accuracy Curve to '{acc_path}'")
    plt.close()

    # =========================================================
    # 2. Plot Loss Curve
    # =========================================================
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Train Loss', color='red', marker='x')

    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    loss_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_path)
    print(f"[INFO] Saved Loss Curve to '{loss_path}'\n")
    plt.close()