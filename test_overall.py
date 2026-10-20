"""
===============================================================================
File        : test_overall.py
Project     : Handwritten Digit Recognition using CNN
Description :
    Loads the trained model weights (best_model.pth) and evaluates
    its overall accuracy on the entire test dataset.
===============================================================================
"""

# =========================================================
# Imports
# =========================================================
import os
import torch

from dataset import get_dataloader
from model import CNN

# =========================================================
# Configuration
# =========================================================

# Automatically select GPU if available, else use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved best model weights
LOAD_PATH = os.path.join("weights", "best_model.pth")

# =========================================================
# Main Evaluation Function
# =========================================================
def test_model():
    print("=" * 70)
    print(" EVALUATING MODEL ON TEST DATASET ")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Check Weights File
    # -----------------------------------------------------
    if not os.path.exists(LOAD_PATH):
        print(f"[ERROR] Weights file not found at: '{LOAD_PATH}'")
        print("Please run 'train.py' first to train and save the model.")
        print("=" * 70)
        return

    # -----------------------------------------------------
    # 2. Load Dataloader
    # -----------------------------------------------------
    _, test_loader = get_dataloader()

    # -----------------------------------------------------
    # 3. Initialize Model & Load Weights
    # -----------------------------------------------------
    model = CNN().to(DEVICE)

    # Load the saved state_dict into the model architecture
   
    model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE, weights_only=True))

    # Set the model to evaluation mode
    
    model.eval()

    print(f"[INFO] Successfully loaded weights from: '{LOAD_PATH}'")
    print(f"[INFO] Running evaluation on device  : {DEVICE}\n")

    # -----------------------------------------------------
    # 4. Evaluation Loop
    # -----------------------------------------------------
    correct = 0
    total = 0

    # Disable gradient calculation to save memory and speed up computation
    
    with torch.no_grad():
        for images, labels in test_loader:

            # Move tensors to the configured device
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass: Get predictions from the model
            outputs = model(images)

            # Extract the class with the highest probability
            
            _, predicted = torch.max(outputs, dim=1)

            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # -----------------------------------------------------
    # 5. Calculate Metrics & Print Results
    # -----------------------------------------------------
    accuracy = 100 * correct / total

    print("-" * 70)
    print(" TEST RESULTS SUMMARY ")
    print("-" * 70)
    print(f"Total test images evaluated : {total:,}")
    print(f"Correctly predicted images  : {correct:,}")
    print(f"Overall Accuracy            : {accuracy:.2f}%")
    print("=" * 70)


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    test_model()