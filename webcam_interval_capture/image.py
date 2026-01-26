"""Image processing utilities."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pillow_avif  # noqa: F401 - registers AVIF codec with Pillow
from numpy.typing import NDArray
from PIL import Image

if TYPE_CHECKING:
    from webcam_interval_capture.config import ImageConfig


def add_timestamp(
    image: NDArray[np.uint8],
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    capture_time: datetime | None = None,
) -> NDArray[np.uint8]:
    """
    Adds a timestamp overlay to the top left of the image.

    Args:
        image: Input BGR image.
        timestamp_format: strftime format string for the timestamp.
        capture_time: The time when the image was captured. If None, uses current time.

    Returns:
        Image with timestamp overlay.
    """
    if capture_time is None:
        capture_time = datetime.now()
    timestamp_text = capture_time.strftime(timestamp_format)
    result = image.copy()

    # Font settings
    position = (20, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # Calculate background rectangle
    text_size = cv2.getTextSize(timestamp_text, font, font_scale, thickness)[0]
    bg_start = (position[0] - 10, position[1] - text_size[1] - 10)
    bg_end = (position[0] + text_size[0] + 10, position[1] + 10)

    # Draw semi-transparent background
    overlay = result.copy()
    cv2.rectangle(overlay, bg_start, bg_end, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

    # Draw text
    cv2.putText(
        result,
        timestamp_text,
        position,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return result


def save_images(
    image: NDArray[np.uint8],
    config: ImageConfig,
    target_height: int,
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    capture_time: datetime | None = None,
) -> tuple[Path, Path, Path]:
    """
    Saves the image as JPEG (resized with timestamp) and AVIF files (original resolution).

    Args:
        image: Full-resolution BGR image (without timestamp).
        config: Image quality configuration.
        target_height: Target height for JPEG (AVIF keeps original resolution).
        timestamp_format: strftime format string for the timestamp overlay.
        capture_time: The time when the image was captured. If None, uses current time.

    Returns:
        Tuple of (current_jpg_path, current_avif_path, timestamped_path).
    """
    # Ensure output directory exists
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filenames
    if capture_time is None:
        capture_time = datetime.now()
    timestamp_str = capture_time.strftime("%Y-%m-%d-%H-%M")
    current_jpg_filename = "webcam_current.jpg"
    current_avif_filename = "webcam_current.avif"
    timestamped_filename = f"webcam_{timestamp_str}.avif"

    current_jpg_path = output_dir / current_jpg_filename
    current_avif_path = output_dir / current_avif_filename
    timestamped_path = output_dir / timestamped_filename

    # Get original dimensions
    h, w = image.shape[:2]

    # Calculate JPEG dimensions (resized to target_height)
    scale = target_height / h
    jpeg_w = int(w * scale)
    jpeg_h = target_height

    # Resize for JPEG
    jpeg_image: NDArray[np.uint8] = cv2.resize(
        image,
        (jpeg_w, jpeg_h),
        interpolation=cv2.INTER_LANCZOS4,
    ).astype(np.uint8)

    # Add timestamp to JPEG only
    jpeg_with_timestamp = add_timestamp(jpeg_image, timestamp_format, capture_time)

    # Convert BGR (OpenCV) to RGB (Pillow)
    jpg_rgb = cv2.cvtColor(jpeg_with_timestamp, cv2.COLOR_BGR2RGB)
    pil_jpg = Image.fromarray(jpg_rgb)

    avif_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_avif = Image.fromarray(avif_rgb)

    # Save current image as optimized JPEG (resized with timestamp)
    pil_jpg.save(
        current_jpg_path,
        "JPEG",
        quality=config.jpeg_quality,
        optimize=True,
    )

    # Save current image as AVIF (original resolution, no timestamp)
    pil_avif.save(
        current_avif_path,
        "AVIF",
        quality=config.avif_quality,
        speed=config.avif_speed,
        subsampling=config.avif_subsampling,
    )

    # Save timestamped history image as AVIF (original resolution, no timestamp overlay)
    pil_avif.save(
        timestamped_path,
        "AVIF",
        quality=config.avif_quality,
        speed=config.avif_speed,
        subsampling=config.avif_subsampling,
    )

    # Print file sizes
    jpg_size = os.path.getsize(current_jpg_path) / 1024
    avif_current_size = os.path.getsize(current_avif_path) / 1024
    avif_history_size = os.path.getsize(timestamped_path) / 1024
    print(
        f"Images saved: {current_jpg_filename} ({jpeg_w}x{jpeg_h}, {jpg_size:.0f}KB), "
        f"{current_avif_filename} ({w}x{h}, {avif_current_size:.0f}KB), "
        f"{timestamped_filename} ({avif_history_size:.0f}KB)"
    )

    return current_jpg_path, current_avif_path, timestamped_path
