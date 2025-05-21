
# Steganography Tool

This is a Python-based GUI application for image steganography using SIFT keypoints and LSB manipulation. It allows users to embed, verify, and detect tampering in watermarked images. Refer to walkthrough.mp4 for a demo of how the app works. I've also uploaded images to use in the app, one watermark image, one cover image, and one image for tampering with to test the tampering detector.

## Features

- Embed watermark images at keypoints using SIFT
- Verify watermark authenticity
- Detect tampering based on watermark consistency


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
