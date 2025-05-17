import cv2  # OpenCV for image processing and SIFT
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk  # For the desktop UI
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # For image handling in the UI
import os
import random


def detect_sift_keypoints(image_path, num_keypoints=None):
    """
    Detect keypoints in an image using SIFT.

    Args:
        image_path: Path to the image
        num_keypoints: Number of keypoints to return (None for all)

    Returns:
        keypoints: List of keypoint objects
        descriptors: Feature descriptors
        image: Original image
    """
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Limit the number of keypoints if specified
    if num_keypoints is not None and len(keypoints) > num_keypoints:
        # Sort keypoints by response strength (higher is better)
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[
            :num_keypoints
        ]
        # Adjust descriptors accordingly
        if descriptors is not None:
            descriptors = descriptors[:num_keypoints]

    return keypoints, descriptors, image


def embed_watermark(cover_image_path, watermark_image_path, output_path, patch_size=5):
    """
    Embed a watermark into an image using SIFT keypoints.

    Args:
        cover_image_path: Path to the cover image
        watermark_image_path: Path to the watermark image
        output_path: Path to save the watermarked image
        patch_size: Size of patch around each keypoint to embed watermark

    Returns:
        watermarked_image: Image with embedded watermark
    """
    # Load watermark and convert to binary
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    _, watermark_binary = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)

    # Detect keypoints
    keypoints, _, cover_image = detect_sift_keypoints(cover_image_path)

    # Get dimensions
    watermark_height, watermark_width = watermark_binary.shape
    cover_height, cover_width = cover_image.shape[:2]

    # Check if watermark is larger than cover image
    if watermark_height > cover_height or watermark_width > cover_width:
        print("Warning: Watermark is larger than cover image.")
        # You could automatically resize here:
        scale_factor = (
            min(cover_height / watermark_height, cover_width / watermark_width) * 0.8
        )  # 80% of max possible size
        new_width = int(watermark_width * scale_factor)
        new_height = int(watermark_height * scale_factor)
        watermark_binary = cv2.resize(
            watermark_binary, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        print(f"Resized watermark to {new_width}x{new_height}")

    # Create a copy of the cover image
    watermarked_image = cover_image.copy()

    # For each keypoint, embed the watermark in the LSB
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        # Define region around keypoint
        x_start = max(0, x - patch_size // 2)
        y_start = max(0, y - patch_size // 2)
        x_end = min(cover_image.shape[1], x + patch_size // 2 + 1)
        y_end = min(cover_image.shape[0], y + patch_size // 2 + 1)

        # For each pixel in the region
        for i in range(y_start, y_end):
            for j in range(x_start, x_end):
                # Get watermark bit (wrap around if watermark is smaller)
                w_i = i - y_start
                w_j = j - x_start
                watermark_bit = watermark_binary[w_i, w_j]

                # Modify LSB of each channel
                for c in range(3):  # For RGB channels
                    # Clear the LSB and set it to the watermark bit
                    watermarked_image[i, j, c] = (
                        watermarked_image[i, j, c] & 0xFE
                    ) | watermark_bit

    # Save the watermarked image
    # cv2.imwrite(output_path, watermarked_image)

    output_path = save_incremented_image(output_path, watermarked_image)

    return watermarked_image, output_path


def extract_watermark(watermarked_image_path, original_watermark_path, patch_size=5):
    """
    Extract watermark from an image using SIFT keypoints.

    Args:
        watermarked_image_path: Path to the potentially watermarked image
        original_watermark_path: Path to the original watermark for comparison
        patch_size: Size of patch around each keypoint

    Returns:
        is_authenticated: Boolean indicating if watermark was found
        extracted_watermarks: list of extracted watermarks at each keypoint
    """
    # Load the watermarked image
    keypoints, _, watermarked_image = detect_sift_keypoints(watermarked_image_path)

    # Load original watermark if provided
    original_watermark = None
    if original_watermark_path:
        original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
        _, orig_watermark_binary = cv2.threshold(
            original_watermark, 127, 1, cv2.THRESH_BINARY
        )

    # Extract watermarks from each keypoint
    # Create an empty watermark image
    extracted_watermark = np.zeros(orig_watermark_binary.shape, dtype=np.uint8)

    # Counter to track how many bits we've filled
    bit_counter = np.zeros(orig_watermark_binary.shape, dtype=int)
    extracted_watermarks = {}

    for idx, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])

        x_start = max(0, x - patch_size // 2)
        y_start = max(0, y - patch_size // 2)
        x_end = min(watermarked_image.shape[1], x + patch_size // 2 + 1)
        y_end = min(watermarked_image.shape[0], y + patch_size // 2 + 1)

        for i in range(y_start, y_end):
            for j in range(x_start, x_end):

                # window
                w_i = i - y_start
                w_j = j - x_start

                # Extract LSB from blue channel (can be any channel)
                r_bit = watermarked_image[i, j, 0] & 1
                g_bit = watermarked_image[i, j, 1] & 1
                b_bit = watermarked_image[i, j, 2] & 1

                # Majority vote
                bit_sum = r_bit + g_bit + b_bit
                watermark_bit = 1 if bit_sum >= 2 else 0  # Majority voting

                # Aggregate watermark bits
                extracted_watermark[w_i, w_j] += watermark_bit
                bit_counter[w_i, w_j] += 1

        extracted_watermarks[idx] = extracted_watermark
    print("extracted WM:", extracted_watermark)
    print("bit counter:", bit_counter)

    # If original watermark is provided, compare extracted watermarks with it
    is_authenticated = False
    if orig_watermark_binary is not None:
        print("orig WM:", orig_watermark_binary)
        # Count how many extracted watermarks match the original
        matching_watermarks = 0
        extract_watermarks_num = len(extracted_watermarks)
        print("length of extracted watermarks:", extract_watermarks_num)
        rand_idx = random.randrange(0, extract_watermarks_num)

        print(f"random watermark, index: {rand_idx}, {extracted_watermarks[rand_idx]}")
        for idx, extracted in extracted_watermarks.items():
            # Resize extracted to match original watermark
            if extracted.shape != orig_watermark_binary.shape:
                extracted_resized = cv2.resize(
                    extracted,
                    (orig_watermark_binary.shape[1], orig_watermark_binary.shape[0]),
                )

            else:
                extracted_resized = extracted

            # Compare
            similarity = np.mean(extracted_resized == orig_watermark_binary)
            if similarity > 0.7:  # If more than 70% similar
                matching_watermarks += 1

        # If more than half of keypoints have matching watermarks, authenticate
        is_authenticated = matching_watermarks > len(keypoints) // 2
        print("matching waterwarks", matching_watermarks)

    return is_authenticated, extracted_watermark


def detect_tampering(image_path, original_watermark_path, patch_size=5):
    """
    Detect if an image has been tampered with based on watermark consistency.

    Args:
        image_path: Path to the potentially tampered image
        original_watermark_path: Path to the original watermark
        patch_size: Size of patch around each keypoint

    Returns:
        is_tampered: Boolean indicating if tampering was detected
        tampered_image: Image with highlighted tampered regions
    """
    # Extract watermarks
    is_authenticated, extracted_watermarks = extract_watermark(
        image_path, original_watermark_path, patch_size
    )

    # Load original watermark
    original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
    _, original_watermark = cv2.threshold(original_watermark, 127, 1, cv2.THRESH_BINARY)

    # Load image for visualization
    keypoints, _, image = detect_sift_keypoints(image_path)
    tampered_image = image.copy()

    # Check each extracted watermark
    inconsistent_keypoints = []
    for idx, (kp, extracted) in enumerate(
        zip(keypoints, extracted_watermarks.values())
    ):
        # Resize extracted to match original watermark
        if extracted.shape != original_watermark.shape:
            extracted_resized = cv2.resize(
                extracted, (original_watermark.shape[1], original_watermark.shape[0])
            )
        else:
            extracted_resized = extracted

        # Compare
        similarity = np.mean(extracted_resized == original_watermark)
        if similarity < 0.7:  # If less than 70% similar
            inconsistent_keypoints.append(kp)

    # Highlight inconsistent keypoints
    tampered_image = cv2.drawKeypoints(
        tampered_image,
        inconsistent_keypoints,
        None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    is_tampered = len(inconsistent_keypoints) > 0

    return is_tampered, tampered_image


def save_incremented_image(output_dir, image):
    base_name = "watermark_file"
    ext = ".jpg"
    i = 1

    # Find next available filename
    while os.path.exists(os.path.join(output_dir, f"{base_name}{i}{ext}")):
        i += 1

    output_path = os.path.join(output_dir, f"{base_name}{i}{ext}")
    cv2.imwrite(output_path, image)

    return output_path
