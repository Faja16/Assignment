import cv2
import numpy as np
import os


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
    watermark_small = cv2.resize(
        watermark, (patch_size, patch_size), interpolation=cv2.INTER_AREA
    )
    _, watermark_binary = cv2.threshold(
        watermark_small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    watermark_binary = watermark_binary.astype(np.uint8)

    # Detect keypoints
    keypoints, _, cover_image = detect_sift_keypoints(cover_image_path, 1000)

    # Create a copy of the cover image
    watermarked_image = cover_image.copy()
    mask = np.zeros_like(watermarked_image, dtype=np.uint8)

    # For each keypoint, embed the watermark in the LSB

    for kp in keypoints:

        x, y = int(kp.pt[0]), int(kp.pt[1])
        half = patch_size // 2

        x_start = max(0, x - half)
        y_start = max(0, y - half)
        x_end = x + half + 1
        y_end = y + half + 1
        # top left and right coordinates of each patch

        # Ensure patch stays inside bounds
        if (
            x - half < 0
            or y - half < 0
            or x + half >= watermarked_image.shape[1]
            or y + half >= watermarked_image.shape[0]
        ):
            continue

        if np.any(mask[y_start:y_end, x_start:x_end]):
            continue

        for i in range(patch_size):
            for j in range(patch_size):
                wm_bit = watermark_binary[i, j]
                for c in range(3):  # RGB
                    if np.any(mask[y_start + i, x_start + j, c]):
                        continue
                    pixel_val = watermarked_image[y_start + i, x_start + j, c]
                    watermarked_image[y_start + i, x_start + j, c] = (
                        pixel_val & 0xFE  # clears LSB
                    ) | wm_bit  # sets LSB

        # Mark patch area as written
        mask[y_start:y_end, x_start:x_end] = 1

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
    keypoints, _, watermarked_image = detect_sift_keypoints(
        watermarked_image_path, 1000
    )

    # Load and threshold the original 3x3 watermark if given
    if original_watermark_path:
        original = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
        # Resize to 3x3
        watermark_small = cv2.resize(
            original, (patch_size, patch_size), interpolation=cv2.INTER_AREA
        )
        _, watermark_binary = cv2.threshold(
            watermark_small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        watermark_binary = watermark_binary.astype(np.uint8)

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

        # Extract LSBs via majority vote across RGB channels
        binary_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
        for i in range(patch_size):
            for j in range(patch_size):
                bit_sum = 0
                for c in range(3):  # iterating through RGB channels
                    pixel_val = watermarked_image[y_start + i, x_start + j, c]
                    bit_sum += pixel_val & 1  # extracts LSB
                bit = 1 if bit_sum >= 2 else 0  # majority vote
                binary_patch[i, j] = bit

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
                    interpolation=cv2.INTER_AREA,
                )

            bit_similarity = np.sum(patch == watermark_binary) / patch.size

            if bit_similarity == 1:
                match_count += 1

        is_authenticated = (
            match_count >= len(keypoints) // 2
        )  # majority out of keypoints

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
        watermark_small = cv2.resize(
            original, (patch_size, patch_size), interpolation=cv2.INTER_AREA
        )
        _, watermark_binary = cv2.threshold(
            watermark_small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        watermark_binary = watermark_binary.astype(np.uint8)

    # Load image for visualization
    keypoints, _, image = detect_sift_keypoints(image_path, 1000)
    tampered_image = image.copy()

    # Check each extracted watermark
    inconsistent_keypoints = []
    for kp, patch in zip(keypoints, extracted_watermarks.values()):
        if patch.shape != watermark_binary.shape:
            patch = cv2.resize(
                patch,
                (watermark_small.shape[1], watermark_small.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        bit_similarity = np.sum(patch == watermark_binary) / patch.size

        if bit_similarity < 1:
            inconsistent_keypoints.append(kp)

    # Highlight inconsistent keypoints
    tampered_image = cv2.drawKeypoints(
        tampered_image,
        inconsistent_keypoints,
        None,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    is_tampered = len(inconsistent_keypoints) > (
        len(keypoints) // 2
    )  # if majority of keypoints are inconsistent, it is tampered

    return is_tampered, tampered_image


def save_incremented_image(output_dir, image):
    """
    saves watermark embedded images with incrementing value into provided directory path.

    Args:
        output_dir: Path to output directory
        image: image being saved


    Returns:
        output_path: path of new embedded image

    """
    base_name = "watermark_file"
    ext = ".png"
    i = 1

    # Find next available filename
    while os.path.exists(os.path.join(output_dir, f"{base_name}{i}{ext}")):
        i += 1

    output_path = os.path.join(output_dir, f"{base_name}{i}{ext}")
    cv2.imwrite(output_path, image)

    return output_path
