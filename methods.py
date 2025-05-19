import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity
import math


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


def embed_watermark(cover_image_path, watermark_image_path, output_path, patch_size=3):
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
    watermark_small = cv2.resize(watermark, (3, 3), interpolation=cv2.INTER_AREA)
    _, watermark_binary = cv2.threshold(
        watermark_small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    watermark_binary = watermark_binary.astype(np.uint8)

    # Detect keypoints
    keypoints, _, cover_image = detect_sift_keypoints(cover_image_path, 6)

    # Get dimensions
    wm_height, wm_width = watermark_binary.shape

    # Create a copy of the cover image
    watermarked_image = cover_image.copy()

    # For each keypoint, embed the watermark in the LSB

    for kp in keypoints:

        x, y = int(kp.pt[0]), int(kp.pt[1])
        half = patch_size // 2

        x_start = max(0, x - half)
        y_start = max(0, y - half)
        # x_end = min(cover_image.shape[1], x + half + 1)
        # y_end = min(cover_image.shape[0], y + half + 1)

        # Ensure patch stays inside bounds
        if (
            x - half < 0
            or y - half < 0
            or x + half >= watermarked_image.shape[1]
            or y + half >= watermarked_image.shape[0]
        ):
            continue

        for i in range(patch_size):
            for j in range(patch_size):
                wm_bit = watermark_binary[i, j]
                for c in range(3):  # RGB
                    pixel_val = watermarked_image[y_start + i, x_start + j, c]
                    watermarked_image[y_start + i, x_start + j, c] = (
                        pixel_val & 0xFE  # clears LSB
                    ) | wm_bit  # sets LSB

    # Save the watermarked image
    # cv2.imwrite(output_path, watermarked_image)

    output_path = save_incremented_image(output_path, watermarked_image)

    return watermarked_image, output_path


def extract_watermark(
    watermarked_image_path, original_watermark_path=None, patch_size=3
):
    """
    Extracts 3x3 binary watermark from 4 SIFT keypoints in the image.

    Args:
        watermarked_image_path (str): Path to watermarked image.
        original_watermark_path (str): Optional path to original 3x3 watermark for comparison.
        patch_size (int): Size of patch around each keypoint (default: 3 for 3x3 neighborhood)

    Returns:
        is_authenticated (bool): True if watermark matches expected pattern.
        extracted_watermarks (dict): Dictionary of extracted 3x3 watermark matrices.
    """
    # Detect keypoints in the watermarked image
    keypoints, _, watermarked_image = detect_sift_keypoints(watermarked_image_path, 6)

    if len(keypoints) < 4:
        raise ValueError(
            "Fewer than 4 keypoints detected. Cannot extract all watermarks."
        )

    # Load and threshold the original 3x3 watermark if given
    if original_watermark_path:
        original = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
        # Resize to 3x3
        watermark_small = cv2.resize(original, (3, 3), interpolation=cv2.INTER_AREA)
        _, watermark_binary = cv2.threshold(
            watermark_small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        watermark_binary = watermark_binary.astype(np.uint8)
        # watermark_normalized = watermark_small.astype(np.float32) / 255.0
        # threshold = 0.5
        # watermark_binary = (watermark_normalized > threshold).astype(np.uint8)

        # _, original_binary = cv2.threshold(watermark_small, 127, 1, cv2.THRESH_BINARY)

    # Check if watermark is too uniform (all black or all white)
    ones_count = np.sum(watermark_binary)
    zeros_count = watermark_binary.size - ones_count

    print(f"Watermark pattern: {watermark_binary.flatten()}")
    print(f"Distribution: {ones_count} ones, {zeros_count} zeros")

    # Dictionary to store extracted 3x3 watermark from each keypoint
    extracted_watermarks = {}

    for idx, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        half = patch_size // 2
        x_start = max(0, x - half)
        y_start = max(0, y - half)

        # Ensure patch stays inside bounds
        if (
            x - half < 0
            or y - half < 0
            or x + half >= watermarked_image.shape[1]
            or y + half >= watermarked_image.shape[0]
        ):
            continue

        # Extract 3x3 neighborhood around the keypoint
        # p = watermarked_image[y - half : y + half + 1, x - half : x + half + 1]

        # Extract LSBs via majority vote across RGB channels
        binary_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
        for i in range(patch_size):
            for j in range(patch_size):
                wm_bit = watermark_binary[i, j]
                bit_sum = 0
                for c in range(3):  # RGB
                    pixel_val = watermarked_image[y_start + i, x_start + j, c]
                    bit_sum += pixel_val & 1  # extracts LSB
                bit = 1 if bit_sum >= 2 else 0  # majority vote

                if bit == wm_bit:
                    binary_patch[i, j] = bit
                else:
                    continue

        extracted_watermarks[idx] = binary_patch

    # === Authentication Logic ===
    is_authenticated = False
    if watermark_binary is not None:
        match_count = 0
        for patch in extracted_watermarks.values():
            if patch.shape != watermark_binary.shape:
                patch = cv2.resize(
                    patch,
                    (watermark_small.shape[1], watermark_small.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            print("patch: \n", patch)
            print("binary WM:\n", watermark_binary)

            # print("binary WM: \n", watermark_binary)

            f1_score = eval(patch, watermark_binary)

            # similarity = np.mean(patch == watermark_binary)
            # score = structural_similarity(
            #     watermark_binary, patch, win_size=3, gaussian_weights=True
            # )
            # print("sim score:", score)
            # if score >= 0.66:  # 70%+ similarity
            #     match_count += 1

            # balanced_accuracy = 0.5 * (tpr + tnr)
            # print(f"balanced accuracy: {balanced_accuracy}, F1 score: {f1_score}")
            # if balanced_accuracy >= 0.70 and f1_score >= 0.5:  # 75%+ similarity
            #     match_count += 1

            # if TP >= patch.size // 2:
            #     match_count += 1

            # hamming_distance = np.sum(patch != watermark_binary)
            # similarity = 1 - hamming_distance / 9  # scale from 0 to 1
            # print("sim", similarity)
            # if similarity >= 0.7:  # allow 1 bit mismatch (8/9 match)
            #     match_count += 1

            # likeness = np.sum(patch == watermark_binary)
            # similarity = likeness / patch.size
            # print("sim:", similarity)
            # if similarity >= 0.7:
            #     match_count += 1

            bit_similarity = np.sum(patch == watermark_binary) / patch.size
            print("bit sim", bit_similarity)
            print("f1 score:", f1_score)

            if f1_score is not math.isnan(f1_score):
                if bit_similarity >= 0.7 and f1_score >= 0.8:
                    match_count += 1

        is_authenticated = (
            match_count >= len(keypoints) // 2
        )  # majority out of keypoints
        print("match count", match_count)

    return is_authenticated, extracted_watermarks


def detect_tampering(image_path, original_watermark_path, patch_size=3):
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
    if original_watermark_path:
        original = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
        # Resize to 3x3
        watermark_small = cv2.resize(original, (3, 3), interpolation=cv2.INTER_AREA)
        _, watermark_binary = cv2.threshold(
            watermark_small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        watermark_binary = watermark_binary.astype(np.uint8)

    # Load image for visualization
    keypoints, _, image = detect_sift_keypoints(image_path, 6)
    tampered_image = image.copy()

    # Check each extracted watermark
    inconsistent_keypoints = []
    for idx, (kp, patch) in enumerate(zip(keypoints, extracted_watermarks.values())):
        if patch.shape != watermark_binary.shape:
            patch = cv2.resize(
                patch,
                (watermark_small.shape[1], watermark_small.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        bit_similarity = np.sum(patch == watermark_binary) / patch.size

        f1_score = eval(patch, watermark_binary)

        bit_similarity = np.sum(patch == watermark_binary) / patch.size
        print("bit sim", bit_similarity)
        print("f1 score:", f1_score)

        if f1_score is not math.isnan(f1_score):
            if bit_similarity < 0.7 and f1_score < 0.8:
                inconsistent_keypoints.append(kp)

    # Highlight inconsistent keypoints
    tampered_image = cv2.drawKeypoints(
        tampered_image,
        inconsistent_keypoints,
        None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    print("inc. kp:", len(inconsistent_keypoints))

    is_tampered = len(inconsistent_keypoints) > (
        len(keypoints) // 2
    )  # if majority of keypoints are inconsistent, it is tampered

    return is_tampered, tampered_image


def eval(patch, watermark_binary):
    TP = np.sum((patch == 1) & (watermark_binary == 1))
    FP = np.sum((patch == 1) & (watermark_binary == 0))
    TN = np.sum((patch == 0) & (watermark_binary == 0))
    FN = np.sum((patch == 0) & (watermark_binary == 1))

    P = np.sum(watermark_binary == 1)
    N = np.sum(watermark_binary == 0)

    precsion = TP / (TP + FP)
    recall = TP / (TP + FN) if (TP + FN) else 0

    print("TP:", TP)
    print("FP:", FP)
    print("TN:", TN)
    print("FN:", FN)

    print("precision:", precsion)
    print("recall:", precsion)

    # Calculate F1 score (mean of precision and recall)
    f1_score = 2 * ((precsion * recall) / (precsion + recall))
    return f1_score


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
