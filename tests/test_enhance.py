"""Tests for webcam_interval_capture.enhance module."""

import numpy as np
import pytest

from webcam_interval_capture.enhance import (
    Enhancer,
    compute_auto_strength,
    dwt_decompose,
    dwt_fusion,
    dwt_reconstruct,
    from_lab,
    to_lab,
)


class TestComputeAutoStrength:
    """Tests for the compute_auto_strength function."""

    def test_dark_image_returns_max_strength(self) -> None:
        """Test that a completely dark image returns max_strength."""
        dark_image = np.zeros((100, 100, 3), dtype=np.uint8)
        strength = compute_auto_strength(dark_image, max_strength=0.15)
        assert strength == pytest.approx(0.15, abs=0.01)

    def test_bright_image_returns_low_strength(self) -> None:
        """Test that a completely bright image returns near-zero strength."""
        bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        strength = compute_auto_strength(bright_image, max_strength=0.15)
        assert strength < 0.01

    def test_medium_brightness_returns_medium_strength(self) -> None:
        """Test that medium brightness returns proportional strength."""
        # 50% brightness
        medium_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        strength = compute_auto_strength(medium_image, max_strength=0.15)
        # Should be approximately 0.5 * 0.15 = 0.075
        assert 0.05 < strength < 0.10

    def test_grayscale_image(self) -> None:
        """Test that grayscale images work correctly."""
        gray_image = np.ones((100, 100), dtype=np.uint8) * 64
        strength = compute_auto_strength(gray_image, max_strength=0.2)
        assert strength > 0.1

    def test_custom_max_strength(self) -> None:
        """Test that custom max_strength is respected."""
        dark_image = np.zeros((100, 100, 3), dtype=np.uint8)
        strength = compute_auto_strength(dark_image, max_strength=0.5)
        assert strength == pytest.approx(0.5, abs=0.01)


class TestDWTFunctions:
    """Tests for DWT decomposition and reconstruction."""

    @pytest.fixture
    def test_image(self) -> np.ndarray:
        """Create a test grayscale image."""
        return np.random.rand(64, 64).astype(np.float32) * 255

    def test_decompose_returns_coefficients(self, test_image: np.ndarray) -> None:
        """Test that decomposition returns coefficient list."""
        coeffs = dwt_decompose(test_image, wavelet="db4", levels=3)
        # Should have 4 elements: [cA3, (cH3,cV3,cD3), (cH2,cV2,cD2), (cH1,cV1,cD1)]
        assert len(coeffs) == 4

    def test_reconstruct_recovers_image(self, test_image: np.ndarray) -> None:
        """Test that reconstruction recovers the original image."""
        coeffs = dwt_decompose(test_image, wavelet="db4", levels=3)
        reconstructed = dwt_reconstruct(coeffs, wavelet="db4")

        # Should be close to original (within floating-point precision)
        h, w = test_image.shape
        reconstructed_cropped = reconstructed[:h, :w]
        np.testing.assert_allclose(reconstructed_cropped, test_image, rtol=1e-2)

    def test_different_wavelets(self, test_image: np.ndarray) -> None:
        """Test that different wavelets work."""
        for wavelet in ["db4", "haar", "sym4"]:
            coeffs = dwt_decompose(test_image, wavelet=wavelet, levels=2)
            reconstructed = dwt_reconstruct(coeffs, wavelet=wavelet)
            assert reconstructed is not None


class TestDWTFusion:
    """Tests for DWT-based image fusion."""

    @pytest.fixture
    def day_image(self) -> np.ndarray:
        """Create a test day image (high detail)."""
        img = np.zeros((64, 64), dtype=np.float32)
        # Add some texture
        img[::2, ::2] = 200
        img[1::2, 1::2] = 200
        return img

    @pytest.fixture
    def night_image(self) -> np.ndarray:
        """Create a test night image (low detail, dark)."""
        return np.ones((64, 64), dtype=np.float32) * 50

    @pytest.fixture
    def brightness_mask(self) -> np.ndarray:
        """Create a test brightness mask."""
        return np.ones((64, 64), dtype=np.float32) * 0.8

    def test_fusion_preserves_night_illumination(
        self,
        day_image: np.ndarray,
        night_image: np.ndarray,
        brightness_mask: np.ndarray,
    ) -> None:
        """Test that fusion preserves overall night illumination."""
        fused = dwt_fusion(
            day_image,
            night_image,
            brightness_mask,
            wavelet="db4",
            levels=2,
            strength=0.5,
            fusion_mode="weighted",
        )

        # Mean should be closer to night than day
        night_mean = night_image.mean()
        day_mean = day_image.mean()
        fused_mean = fused.mean()

        # Fused mean should be between night and day, but closer to night
        assert fused_mean < day_mean
        assert abs(fused_mean - night_mean) < abs(fused_mean - day_mean)

    def test_fusion_with_zero_strength(
        self,
        day_image: np.ndarray,
        night_image: np.ndarray,
        brightness_mask: np.ndarray,
    ) -> None:
        """Test that zero strength returns night image."""
        fused = dwt_fusion(
            day_image,
            night_image,
            brightness_mask,
            strength=0.0,
        )

        # Should be close to night image
        h, w = night_image.shape
        fused_cropped = fused[:h, :w]
        np.testing.assert_allclose(fused_cropped, night_image, rtol=0.1)


class TestColorSpaceConversion:
    """Tests for LAB color space conversion."""

    @pytest.fixture
    def test_bgr(self) -> np.ndarray:
        """Create a test BGR image."""
        return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_to_lab_returns_three_channels(self, test_bgr: np.ndarray) -> None:
        """Test that to_lab returns three float32 channels."""
        L, a, b = to_lab(test_bgr)

        assert L.dtype == np.float32
        assert a.dtype == np.float32
        assert b.dtype == np.float32
        assert L.shape == test_bgr.shape[:2]

    def test_from_lab_returns_bgr(self, test_bgr: np.ndarray) -> None:
        """Test that from_lab returns BGR uint8 image."""
        L, a, b = to_lab(test_bgr)
        result = from_lab(L, a, b)

        assert result.dtype == np.uint8
        assert result.shape == test_bgr.shape

    def test_roundtrip_conversion(self, test_bgr: np.ndarray) -> None:
        """Test that LAB roundtrip preserves image."""
        L, a, b = to_lab(test_bgr)
        result = from_lab(L, a, b)

        # Should be close to original (LAB conversion has some precision loss)
        np.testing.assert_allclose(result, test_bgr, atol=25)


class TestEnhancerInit:
    """Tests for Enhancer initialization."""

    def test_default_values(self) -> None:
        """Test that Enhancer has sensible defaults."""
        enhancer = Enhancer()

        assert enhancer.max_strength == 0.15
        assert enhancer.brightness_threshold == 0.3
        assert enhancer.wavelet == "db4"
        assert enhancer.levels == 3
        assert enhancer.fusion_mode == "weighted"

    def test_custom_values(self) -> None:
        """Test that custom values are applied."""
        enhancer = Enhancer(
            max_strength=0.2,
            brightness_threshold=0.5,
            wavelet="haar",
            levels=4,
            fusion_mode="max_energy",
        )

        assert enhancer.max_strength == 0.2
        assert enhancer.brightness_threshold == 0.5
        assert enhancer.wavelet == "haar"
        assert enhancer.levels == 4
        assert enhancer.fusion_mode == "max_energy"


class TestEnhancerEnhance:
    """Tests for Enhancer.enhance() method."""

    @pytest.fixture
    def test_frame(self) -> np.ndarray:
        """Create a test frame (night image)."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 50

    @pytest.fixture
    def test_reference(self) -> np.ndarray:
        """Create a test reference (day image)."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 150
        # Add some detail
        img[::2, ::2, :] = 200
        return img

    def test_enhance_without_reference_returns_original(
        self, test_frame: np.ndarray
    ) -> None:
        """Test that enhance without reference returns original resolution."""
        enhancer = Enhancer()
        result = enhancer.enhance(test_frame, reference=None)

        assert result is not None
        assert result.shape == test_frame.shape

    def test_enhance_with_reference(
        self, test_frame: np.ndarray, test_reference: np.ndarray
    ) -> None:
        """Test that enhance with reference produces enhanced result."""
        enhancer = Enhancer(max_strength=0.15)
        result = enhancer.enhance(test_frame, reference=test_reference)

        assert result is not None
        assert result.shape == test_frame.shape

    def test_enhance_preserves_original_resolution(
        self, test_frame: np.ndarray, test_reference: np.ndarray
    ) -> None:
        """Test that enhance preserves original resolution."""
        enhancer = Enhancer()
        result = enhancer.enhance(test_frame, reference=test_reference)

        assert result is not None
        assert result.shape == test_frame.shape

    def test_enhance_bright_image_skips_enhancement(self) -> None:
        """Test that bright images skip enhancement."""
        bright_frame = np.ones((480, 640, 3), dtype=np.uint8) * 250
        reference = np.ones((480, 640, 3), dtype=np.uint8) * 200

        enhancer = Enhancer(max_strength=0.15)
        result = enhancer.enhance(bright_frame, reference=reference)

        assert result is not None
        # Should return original resolution
        assert result.shape == bright_frame.shape

    def test_enhance_resizes_reference_if_needed(self) -> None:
        """Test that reference is resized to match frame."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        reference = np.ones((720, 1280, 3), dtype=np.uint8) * 150

        enhancer = Enhancer()
        result = enhancer.enhance(frame, reference=reference)

        assert result is not None
        # Output should match frame resolution, not reference
        assert result.shape == frame.shape


class TestEnhancerIntegration:
    """Integration-style tests for Enhancer."""

    def test_full_workflow(self) -> None:
        """Test the full enhancement workflow."""
        # Create test images
        night_image = np.ones((480, 640, 3), dtype=np.uint8) * 40
        day_image = np.ones((480, 640, 3), dtype=np.uint8) * 180
        day_image[::4, ::4, :] = 220  # Add texture

        enhancer = Enhancer(max_strength=0.15)
        result = enhancer.enhance(night_image, reference=day_image)

        assert result is not None
        assert result.shape == (480, 640, 3)  # Original resolution preserved
        assert result.dtype == np.uint8
