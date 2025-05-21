
# Steganography Tool

This is a Python-based GUI application for image steganography using SIFT keypoints and LSB manipulation. It allows users to embed, verify, and detect tampering in watermarked images.

## Features

- Embed watermark images at keypoints using SIFT
- Verify watermark authenticity
- Detect tampering based on watermark consistency
- Rotational watermark embedding for added robustness

## Requirements

Before running the app, make sure the following Python libraries are installed:

- `cv2` (OpenCV)
- `numpy`
- `customtkinter`
- `PIL` (Pillow)
- `matplotlib`

You can install the required packages using:

```bash
pip install opencv-python numpy customtkinter pillow matplotlib


## Run the App

To start the application, use one of the following commands:

```bash
python -m main 

OR

python main.py
Make sure you are in the root directory where main.py is located.