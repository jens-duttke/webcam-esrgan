"""Tests for webcam_esrgan.enhance module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from webcam_esrgan.enhance import Enhancer, _apply_torchvision_workaround


class TestTorchvisionWorkaround:
    """Tests for the torchvision compatibility workaround."""

    def test_adds_rgb_to_grayscale_attribute(self) -> None:
        """Test that workaround adds rgb_to_grayscale if missing."""
        mock_F_tv = MagicMock()
        mock_F_tv.to_grayscale = MagicMock()
        del mock_F_tv.rgb_to_grayscale  # Simulate missing attribute

        with patch.dict(
            sys.modules,
            {"torchvision.transforms.functional": mock_F_tv},
        ):
            _apply_torchvision_workaround()

            # The workaround should add rgb_to_grayscale
            assert mock_F_tv.rgb_to_grayscale == mock_F_tv.to_grayscale


class TestEnhancerInit:
    """Tests for Enhancer initialization."""

    def test_default_values(self) -> None:
        """Test that Enhancer has sensible defaults."""
        enhancer = Enhancer()

        assert enhancer.target_height == 1080
        assert enhancer.upscale_factor == 3
        assert enhancer.enhancement_blend == 0.8
        assert enhancer.weights_dir == Path.cwd() / "weights"

    def test_custom_values(self) -> None:
        """Test that custom values are applied."""
        enhancer = Enhancer(
            target_height=720,
            upscale_factor=2,
            enhancement_blend=0.5,
            weights_dir=Path("/custom/weights"),
        )

        assert enhancer.target_height == 720
        assert enhancer.upscale_factor == 2
        assert enhancer.enhancement_blend == 0.5
        assert enhancer.weights_dir == Path("/custom/weights")

    def test_is_initialized_false_by_default(self) -> None:
        """Test that enhancer is not initialized by default."""
        enhancer = Enhancer()

        assert enhancer.is_initialized is False

    def test_model_url_and_filename(self) -> None:
        """Test that model URL and filename are set correctly."""
        assert Enhancer.MODEL_FILENAME == "realesr-general-x4v3.pth"
        assert "realesr-general-x4v3.pth" in Enhancer.MODEL_URL


class TestEnhancerInitialize:
    """Tests for Enhancer.initialize() method."""

    @pytest.fixture
    def enhancer(self, tmp_path: Path) -> Enhancer:
        """Create an enhancer with temp weights directory."""
        return Enhancer(weights_dir=tmp_path)

    def test_initialize_import_error(self, enhancer: Enhancer) -> None:
        """Test initialization fails gracefully when realesrgan not installed."""
        # Remove the module if it exists
        with (
            patch.dict(sys.modules, {"realesrgan": None}),
            patch(
                "webcam_esrgan.enhance._apply_torchvision_workaround",
                side_effect=ImportError("No module named 'realesrgan'"),
            ),
        ):
            result = enhancer.initialize()

            assert result is False
            assert enhancer.is_initialized is False

    def test_initialize_success(self, enhancer: Enhancer) -> None:
        """Test successful initialization with mocked dependencies."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_model = MagicMock()
        mock_srvgg = MagicMock(return_value=mock_model)

        mock_upsampler = MagicMock()
        mock_realesrgan = MagicMock()
        mock_realesrgan.RealESRGANer.return_value = mock_upsampler

        # Create a fake model file
        model_path = enhancer.weights_dir / Enhancer.MODEL_FILENAME
        enhancer.weights_dir.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"fake model data")

        with (
            patch("webcam_esrgan.enhance._apply_torchvision_workaround"),
            patch.dict(
                sys.modules,
                {
                    "realesrgan": mock_realesrgan,
                    "realesrgan.archs.srvgg_arch": MagicMock(
                        SRVGGNetCompact=mock_srvgg
                    ),
                    "torch": mock_torch,
                },
            ),
            patch(
                "webcam_esrgan.enhance.RealESRGANer",
                mock_realesrgan.RealESRGANer,
                create=True,
            ),
        ):
            result = enhancer.initialize()

            assert result is True
            assert enhancer.is_initialized is True

    def test_initialize_downloads_model_if_missing(self, enhancer: Enhancer) -> None:
        """Test that model is downloaded if not present."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_urlretrieve = MagicMock()

        with (
            patch("webcam_esrgan.enhance._apply_torchvision_workaround"),
            patch("urllib.request.urlretrieve", mock_urlretrieve),
            patch.dict(
                sys.modules,
                {
                    "realesrgan": MagicMock(),
                    "realesrgan.archs.srvgg_arch": MagicMock(),
                    "torch": mock_torch,
                },
            ),
        ):
            # Model file doesn't exist, should trigger download
            enhancer.initialize()

            mock_urlretrieve.assert_called_once()
            call_args = mock_urlretrieve.call_args[0]
            assert call_args[0] == Enhancer.MODEL_URL

    def test_initialize_with_cuda(self, enhancer: Enhancer) -> None:
        """Test initialization detects CUDA."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"

        mock_upsampler = MagicMock()
        mock_realesrgan = MagicMock()
        mock_realesrgan.RealESRGANer.return_value = mock_upsampler

        model_path = enhancer.weights_dir / Enhancer.MODEL_FILENAME
        enhancer.weights_dir.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"fake model data")

        with (
            patch("webcam_esrgan.enhance._apply_torchvision_workaround"),
            patch.dict(
                sys.modules,
                {
                    "realesrgan": mock_realesrgan,
                    "realesrgan.archs.srvgg_arch": MagicMock(),
                    "torch": mock_torch,
                },
            ),
        ):
            enhancer.initialize()

            # Check that RealESRGANer was called with GPU settings
            call_kwargs = mock_realesrgan.RealESRGANer.call_args[1]
            assert call_kwargs["device"] == "cuda"
            assert call_kwargs["half"] is True

    def test_initialize_general_exception(self, enhancer: Enhancer) -> None:
        """Test initialization handles general exceptions."""
        with patch(
            "webcam_esrgan.enhance._apply_torchvision_workaround",
            side_effect=RuntimeError("Something went wrong"),
        ):
            result = enhancer.initialize()

            assert result is False
            assert enhancer.is_initialized is False


class TestEnhancerEnhance:
    """Tests for Enhancer.enhance() method."""

    @pytest.fixture
    def test_image(self) -> np.ndarray:
        """Create a test image."""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_enhance_returns_none_when_not_initialized(
        self, test_image: np.ndarray
    ) -> None:
        """Test that enhance returns None if not initialized."""
        enhancer = Enhancer()

        result = enhancer.enhance(test_image)

        assert result is None

    def test_enhance_success(self, test_image: np.ndarray) -> None:
        """Test successful enhancement."""
        enhancer = Enhancer(target_height=240, upscale_factor=2)

        # Create a mock upsampler
        mock_upsampler = MagicMock()
        # Return a properly sized RGB image (after 2x upscale)
        enhanced_h = 240  # Target height
        enhanced_w = 320
        mock_enhanced = np.random.randint(
            0, 256, (enhanced_h, enhanced_w, 3), dtype=np.uint8
        )
        mock_upsampler.enhance.return_value = (mock_enhanced, None)
        enhancer._upsampler = mock_upsampler

        result = enhancer.enhance(test_image)

        assert result is not None
        assert result.shape[0] == 240  # Target height
        assert result.dtype == np.uint8

    def test_enhance_resizes_to_target_height(self) -> None:
        """Test that output is resized when it doesn't match target height."""
        enhancer = Enhancer(target_height=480, upscale_factor=2)

        # Input image
        test_image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)

        # Mock upsampler returns wrong height (needs resizing)
        mock_upsampler = MagicMock()
        mock_enhanced = np.random.randint(
            0, 256, (400, 600, 3), dtype=np.uint8
        )  # Not 480!
        mock_upsampler.enhance.return_value = (mock_enhanced, None)
        enhancer._upsampler = mock_upsampler

        result = enhancer.enhance(test_image)

        assert result is not None
        # Should be resized to exactly 480
        assert result.shape[0] == 480

    def test_enhance_with_pre_shrink(self) -> None:
        """Test that large images are pre-shrunk before enhancement."""
        # Create a large image (larger than target_height / upscale_factor)
        large_image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

        enhancer = Enhancer(target_height=720, upscale_factor=3)
        # target_pre_h = 720 / 3 = 240, so 1080 > 240 → pre-shrink

        mock_upsampler = MagicMock()
        mock_enhanced = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        mock_upsampler.enhance.return_value = (mock_enhanced, None)
        enhancer._upsampler = mock_upsampler

        result = enhancer.enhance(large_image)

        assert result is not None
        # Verify upsampler received a smaller image (pre-shrunk)
        call_args = mock_upsampler.enhance.call_args[0][0]
        assert call_args.shape[0] < large_image.shape[0]

    def test_enhance_without_pre_shrink(self) -> None:
        """Test that small images are NOT pre-shrunk."""
        # Create a small image (smaller than target_height / upscale_factor)
        small_image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)

        enhancer = Enhancer(target_height=720, upscale_factor=3)
        # target_pre_h = 720 / 3 = 240, so 200 < 240 → no pre-shrink

        mock_upsampler = MagicMock()
        mock_enhanced = np.random.randint(0, 256, (600, 900, 3), dtype=np.uint8)
        mock_upsampler.enhance.return_value = (mock_enhanced, None)
        enhancer._upsampler = mock_upsampler

        result = enhancer.enhance(small_image)

        assert result is not None
        # Verify upsampler received original size (converted to RGB)
        call_args = mock_upsampler.enhance.call_args[0][0]
        assert call_args.shape[0] == small_image.shape[0]

    def test_enhance_blends_with_original(self) -> None:
        """Test that enhancement blends with original when blend < 1."""
        enhancer = Enhancer(
            target_height=240,
            upscale_factor=2,
            enhancement_blend=0.5,  # 50/50 blend
        )

        test_image = np.ones((120, 160, 3), dtype=np.uint8) * 100

        mock_upsampler = MagicMock()
        # Return a different colored image
        mock_enhanced = np.ones((240, 320, 3), dtype=np.uint8) * 200
        mock_upsampler.enhance.return_value = (mock_enhanced, None)
        enhancer._upsampler = mock_upsampler

        result = enhancer.enhance(test_image)

        assert result is not None
        # With 50/50 blend, result should be between original and enhanced
        # (approximately, due to resize interpolation)
        mean_value = result.mean()
        assert 100 < mean_value < 200

    def test_enhance_fallback_on_error(self, test_image: np.ndarray) -> None:
        """Test that enhance falls back to resize on error."""
        enhancer = Enhancer(target_height=240, upscale_factor=2)

        mock_upsampler = MagicMock()
        mock_upsampler.enhance.side_effect = RuntimeError("GPU error")
        enhancer._upsampler = mock_upsampler

        result = enhancer.enhance(test_image)

        # Should return resized image as fallback
        assert result is not None
        assert result.shape[0] == 240

    def test_enhance_no_blend_when_blend_is_one(self) -> None:
        """Test that no blending occurs when enhancement_blend is 1.0."""
        enhancer = Enhancer(
            target_height=240,
            upscale_factor=2,
            enhancement_blend=1.0,
        )

        test_image = np.zeros((120, 160, 3), dtype=np.uint8)

        mock_upsampler = MagicMock()
        mock_enhanced = np.ones((240, 320, 3), dtype=np.uint8) * 255
        mock_upsampler.enhance.return_value = (mock_enhanced, None)
        enhancer._upsampler = mock_upsampler

        result = enhancer.enhance(test_image)

        assert result is not None
        # Should be all white (enhanced only)
        assert result.mean() > 250


class TestEnhancerIntegration:
    """Integration-style tests for Enhancer."""

    def test_full_workflow_without_model(self) -> None:
        """Test the full workflow fails gracefully without the model."""
        enhancer = Enhancer()

        # Initialize will fail because realesrgan is not installed
        # (or we mock it to fail)
        with patch(
            "webcam_esrgan.enhance._apply_torchvision_workaround",
            side_effect=ImportError("No realesrgan"),
        ):
            result = enhancer.initialize()
            assert result is False

        # Enhance should return None
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = enhancer.enhance(test_image)
        assert result is None
