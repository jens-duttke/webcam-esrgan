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
    from webcam_esrgan.config import ImageConfig


def add_timestamp(
    image: NDArray[np.uint8],
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
) -> NDArray[np.uint8]:
    """
    Adds a timestamp overlay to the top left of the image.

    Args:
        image: Input BGR image.
        timestamp_format: strftime format string for the timestamp.

    Returns:
        Image with timestamp overlay.
    """
    timestamp_text = datetime.now().strftime(timestamp_format)
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
    image_with_timestamp: NDArray[np.uint8],
    image_without_timestamp: NDArray[np.uint8],
    config: ImageConfig,
) -> tuple[Path, Path, Path]:
    """
    Saves the image as JPEG (with timestamp) and AVIF files (without timestamp).

    Args:
        image_with_timestamp: Image with timestamp overlay (for JPEG).
        image_without_timestamp: Image without timestamp overlay (for AVIF files).
        config: Image quality configuration.

    Returns:
        Tuple of (current_jpg_path, current_avif_path, timestamped_path).
    """
    # Ensure output directory exists
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filenames
    timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    current_jpg_filename = "webcam_current.jpg"
    current_avif_filename = "webcam_current.avif"
    timestamped_filename = f"webcam_{timestamp_str}.avif"

    current_jpg_path = output_dir / current_jpg_filename
    current_avif_path = output_dir / current_avif_filename
    timestamped_path = output_dir / timestamped_filename

    # Convert BGR (OpenCV) to RGB (Pillow)
    jpg_rgb = cv2.cvtColor(image_with_timestamp, cv2.COLOR_BGR2RGB)
    pil_jpg = Image.fromarray(jpg_rgb)

    avif_rgb = cv2.cvtColor(image_without_timestamp, cv2.COLOR_BGR2RGB)
    pil_avif = Image.fromarray(avif_rgb)

    # Save current image as optimized JPEG (with timestamp)
    pil_jpg.save(
        current_jpg_path,
        "JPEG",
        quality=config.jpeg_quality,
        optimize=True,
    )

    # Save current image as AVIF (without timestamp)
    pil_avif.save(
        current_avif_path,
        "AVIF",
        quality=config.avif_quality,
        speed=config.avif_speed,
        subsampling=config.avif_subsampling,
    )

    # Save timestamped history image as AVIF (without timestamp overlay)
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
        f"Images saved: {current_jpg_filename} ({jpg_size:.0f}KB), "
        f"{current_avif_filename} ({avif_current_size:.0f}KB), "
        f"{timestamped_filename} ({avif_history_size:.0f}KB)"
    )

    return current_jpg_path, current_avif_path, timestamped_path
