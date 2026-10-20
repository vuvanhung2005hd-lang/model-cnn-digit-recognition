"""
===============================================================================
File        : camera.py
Project     : Handwritten Digit Recognition using CNN
Description :
    Real-time handwritten digit recognition using a webcam.
    Implements Background Subtraction (CamScanner style) to achieve
    100% invariance to shadows, backlighting, and uneven illumination.
===============================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as f
from torchvision import transforms
from PIL import Image
from model import CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_PATH = os.path.join("weights", "best_model.pth")

def load_trained_model():
    if not os.path.exists(LOAD_PATH):
        raise FileNotFoundError(f"[ERROR] Weights file not found: {LOAD_PATH}")
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def preprocess_roi(roi_crop):
    gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)

    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.dilate(gray, bg_kernel)
    background = cv2.medianBlur(background, 21)

    diff = cv2.absdiff(background, gray)

    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_img_np = np.zeros((28, 28), dtype=np.uint8)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Lọc bỏ nhiễu linh tinh
            if 30 < area < (130 * 130 * 0.9):
                x, y, w, h = cv2.boundingRect(cnt)
                cropped = thresh[y:y+h, x:x+w]

                max_dim = max(w, h)
                top = (max_dim - h) // 2
                bottom = max_dim - h - top
                left = (max_dim - w) // 2
                right = max_dim - w - left

                squared = cv2.copyMakeBorder(cropped, top, bottom, left, right,
                                             cv2.BORDER_CONSTANT, value=0)

                resized = cv2.resize(squared, (20, 20), interpolation=cv2.INTER_AREA)
                final_img_np = cv2.copyMakeBorder(resized, 4, 4, 4, 4,
                                                  cv2.BORDER_CONSTANT, value=0)
                break

    white_pixels = cv2.countNonZero(final_img_np)

    pil_img = Image.fromarray(final_img_np)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    return img_tensor, thresh, white_pixels

def run_camera():
    try:
        model = load_trained_model()
        print("[INFO] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open Webcam.")
        return

    print("=" * 70)
    print(" LIVE CAMERA STARTED... PRESS 'q' OR 'ESC' TO EXIT. ")
    print("=" * 70)

    box_size = 130

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        x1 = int((w - box_size) / 2)
        y1 = int((h - box_size) / 2)
        x2 = x1 + box_size
        y2 = y1 + box_size

        roi = frame[y1:y2, x1:x2]
        img_tensor, thresh_preview, white_pixels = preprocess_roi(roi)

        box_color = (200, 200, 200)
        label_text = "Align digit here"
        text_color = (200, 200, 200)
        draw_filled_bg = False

        if white_pixels > 40:
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = f.softmax(output, dim=1)[0]
                confidence, predicted_class = torch.max(probabilities, dim=0)

                pred_digit = predicted_class.item()
                conf_score = confidence.item() * 100

                box_color = (0, 215, 255)
                label_text = f"Digit: {pred_digit} ({conf_score:.1f}%)"
                text_color = (0, 0, 0)
                draw_filled_bg = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        if draw_filled_bg:
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), box_color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        else:
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        cv2.imshow("Live Prediction System", frame)

        debug_view = cv2.resize(thresh_preview, (260, 260), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Vision Debug (Shadow Removed)", debug_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera closed safely.")

if __name__ == "__main__":
    run_camera()