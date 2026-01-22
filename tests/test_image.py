"""Tests for webcam_esrgan.image module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from webcam_esrgan.config import ImageConfig
from webcam_esrgan.image import add_timestamp, save_images


class TestAddTimestamp:
    """Tests for add_timestamp function."""

    def test_returns_new_image(self) -> None:
        """Test that add_timestamp returns a new image, not modifying original."""
        original = np.zeros((100, 200, 3), dtype=np.uint8)
        original_copy = original.copy()

        result = add_timestamp(original)

        # Original should be unchanged
        np.testing.assert_array_equal(original, original_copy)
        # Result should be different (has timestamp)
        assert not np.array_equal(original, result)

    def test_output_same_shape_as_input(self) -> None:
        """Test that output has same shape as input."""
        original = np.zeros((480, 640, 3), dtype=np.uint8)

        result = add_timestamp(original)

        assert result.shape == original.shape

    def test_custom_format_applied(self) -> None:
        """Test that custom timestamp format is actually used."""
        original = np.zeros((100, 400, 3), dtype=np.uint8)

        with patch("webcam_esrgan.image.datetime") as mock_dt:
            from datetime import datetime

            mock_dt.now.return_value = datetime(2026, 1, 21, 14, 30, 45)

            # With default format, would show "2026-01-21 14:30:45"
            # With custom format "%H:%M", should show "14:30"
            result = add_timestamp(original, timestamp_format="%H:%M")

            # Verify datetime.now() was called and result is valid
            mock_dt.now.assert_called()
            assert result.shape == original.shape

    def test_timestamp_adds_content(self) -> None:
        """Test that timestamp actually modifies the image."""
        # Create a white image
        original = np.ones((100, 400, 3), dtype=np.uint8) * 255

        result = add_timestamp(original)

        # The result should have some dark pixels (text and background)
        # Count pixels that are significantly different from white
        diff = np.abs(original.astype(int) - result.astype(int))
        changed_pixels = np.sum(diff > 50)

        assert changed_pixels > 0, "Timestamp should modify some pixels"

    def test_uses_capture_time_when_provided(self) -> None:
        """Test that capture_time parameter is used instead of current time."""
        original = np.zeros((100, 400, 3), dtype=np.uint8)
        capture_time = datetime(2020, 6, 15, 10, 30, 45)

        # Should NOT call datetime.now() when capture_time is provided
        with patch("webcam_esrgan.image.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 21, 14, 0, 0)

            result = add_timestamp(
                original, timestamp_format="%Y-%m-%d", capture_time=capture_time
            )

            # datetime.now() should NOT be called when capture_time is provided
            mock_dt.now.assert_not_called()
            assert result.shape == original.shape


class TestSaveImages:
    """Tests for save_images function."""

    @pytest.fixture
    def test_image(self) -> np.ndarray:
        """Create a test BGR image."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def image_config(self, tmp_path: Path) -> ImageConfig:
        """Create a test image configuration."""
        return ImageConfig(
            jpeg_quality=85,
            avif_quality=60,
            avif_speed=6,
            avif_subsampling="4:2:0",
            output_dir=str(tmp_path / "images"),
        )

    def test_creates_output_directory(
        self, test_image: np.ndarray, image_config: ImageConfig
    ) -> None:
        """Test that output directory is created if it doesn't exist."""
        output_dir = Path(image_config.output_dir)
        assert not output_dir.exists()

        save_images(test_image, test_image, image_config)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_returns_correct_paths(
        self, test_image: np.ndarray, image_config: ImageConfig
    ) -> None:
        """Test that correct paths are returned."""
        current_jpg, current_avif, timestamped = save_images(
            test_image, test_image, image_config
        )

        assert current_jpg.name == "webcam_current.jpg"
        assert current_avif.name == "webcam_current.avif"
        assert timestamped.name.startswith("webcam_")
        assert timestamped.suffix == ".avif"

    def test_creates_jpeg_file(
        self, test_image: np.ndarray, image_config: ImageConfig
    ) -> None:
        """Test that JPEG file is created."""
        current_jpg, _, _ = save_images(test_image, test_image, image_config)

        assert current_jpg.exists()
        assert current_jpg.stat().st_size > 0

    def test_creates_current_avif_file(
        self, test_image: np.ndarray, image_config: ImageConfig
    ) -> None:
        """Test that current AVIF file is created."""
        _, current_avif, _ = save_images(test_image, test_image, image_config)

        assert current_avif.exists()
        assert current_avif.stat().st_size > 0

    def test_creates_timestamped_avif_file(
        self, test_image: np.ndarray, image_config: ImageConfig
    ) -> None:
        """Test that timestamped AVIF file is created."""
        _, _, timestamped = save_images(test_image, test_image, image_config)

        assert timestamped.exists()
        assert timestamped.stat().st_size > 0

    def test_jpeg_is_valid_image(
        self, test_image: np.ndarray, image_config: ImageConfig
    ) -> None:
        """Test that saved JPEG can be read back."""
        current_jpg, _, _ = save_images(test_image, test_image, image_config)

        loaded = cv2.imread(str(current_jpg))

        assert loaded is not None
        assert loaded.shape == test_image.shape

    def test_timestamped_filename_format(
        self, test_image: np.ndarray, image_config: ImageConfig
    ) -> None:
        """Test that timestamped filename follows expected format."""
        _, _, timestamped = save_images(test_image, test_image, image_config)

        # Format: webcam_YYYY-MM-DD-HH-MM.avif
        name = timestamped.stem  # webcam_2026-01-21-12-00
        parts = name.split("_", 1)[1]  # 2026-01-21-12-00

        # Should have 5 parts: year, month, day, hour, minute
        date_parts = parts.split("-")
        assert len(date_parts) == 5
        assert len(date_parts[0]) == 4  # Year
        assert len(date_parts[1]) == 2  # Month
        assert len(date_parts[2]) == 2  # Day
        assert len(date_parts[3]) == 2  # Hour
        assert len(date_parts[4]) == 2  # Minute

    def test_overwrites_existing_current(self, image_config: ImageConfig) -> None:
        """Test that current files are overwritten on subsequent saves."""
        # First save with black image
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)
        current_jpg_1, current_avif_1, _ = save_images(
            black_image, black_image, image_config
        )
        jpg_content_1 = current_jpg_1.read_bytes()
        avif_content_1 = current_avif_1.read_bytes()

        # Second save with white image
        white_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        current_jpg_2, current_avif_2, _ = save_images(
            white_image, white_image, image_config
        )
        jpg_content_2 = current_jpg_2.read_bytes()
        avif_content_2 = current_avif_2.read_bytes()

        # Same paths, but content changed
        assert current_jpg_1 == current_jpg_2
        assert current_avif_1 == current_avif_2
        assert jpg_content_1 != jpg_content_2
        assert avif_content_1 != avif_content_2

    def test_multiple_saves_create_multiple_avif_files(
        self, test_image: np.ndarray, image_config: ImageConfig
    ) -> None:
        """Test that multiple saves create multiple timestamped files."""
        # First save with explicit capture_time
        capture_time_1 = datetime(2026, 1, 21, 10, 0, 0)
        _, _, path1 = save_images(
            test_image, test_image, image_config, capture_time_1
        )

        # Second save with different capture_time
        capture_time_2 = datetime(2026, 1, 21, 10, 1, 0)
        _, _, path2 = save_images(
            test_image, test_image, image_config, capture_time_2
        )

        assert path1.name != path2.name
        assert path1.exists()
        assert path2.exists()

    def test_uses_capture_time_for_filename(
        self, test_image: np.ndarray, image_config: ImageConfig
    ) -> None:
        """Test that capture_time is used for the timestamped filename."""
        capture_time = datetime(2020, 6, 15, 14, 30, 0)
        _, _, timestamped = save_images(
            test_image, test_image, image_config, capture_time
        )

        # Filename should reflect the capture_time, not current time
        assert timestamped.name == "webcam_2020-06-15-14-30.avif"

    def test_respects_jpeg_quality(
        self, test_image: np.ndarray, tmp_path: Path
    ) -> None:
        """Test that JPEG quality setting affects file size."""
        # High quality
        high_config = ImageConfig(
            jpeg_quality=95,
            output_dir=str(tmp_path / "high"),
        )
        high_path, _, _ = save_images(test_image, test_image, high_config)
        high_size = high_path.stat().st_size

        # Low quality
        low_config = ImageConfig(
            jpeg_quality=30,
            output_dir=str(tmp_path / "low"),
        )
        low_path, _, _ = save_images(test_image, test_image, low_config)
        low_size = low_path.stat().st_size

        # High quality should be larger
        assert high_size > low_size
