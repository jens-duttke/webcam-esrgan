"""Tests for webcam_esrgan.camera module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from webcam_esrgan.camera import Camera
from webcam_esrgan.config import CameraConfig


class TestCamera:
    """Tests for Camera class."""

    @pytest.fixture
    def camera_config(self) -> CameraConfig:
        """Create a test camera configuration."""
        return CameraConfig(
            ip="192.168.1.100",
            user="admin",
            password="secret",
            channel="0",
        )

    @pytest.fixture
    def camera(self, camera_config: CameraConfig) -> Camera:
        """Create a test camera instance."""
        return Camera(camera_config)

    def test_snapshot_url_set_correctly(self, camera: Camera) -> None:
        """Test that snapshot URL is correctly set from config."""
        assert "192.168.1.100" in camera.snapshot_url
        assert "admin" in camera.snapshot_url
        assert "channel=0" in camera.snapshot_url

    def test_ssl_context_ignores_certificates(self, camera: Camera) -> None:
        """Test that SSL context is configured to ignore certificate errors."""
        import ssl

        assert camera._ssl_context.check_hostname is False
        assert camera._ssl_context.verify_mode == ssl.CERT_NONE

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_fetch_snapshot_success(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test successful snapshot fetch."""
        # Create a minimal valid JPEG
        import cv2

        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode(".jpg", test_image)

        # Mock the response
        mock_response = MagicMock()
        mock_response.read.return_value = jpeg_bytes.tobytes()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = camera.fetch_snapshot()

        assert result is not None
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_fetch_snapshot_returns_none_on_error(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test that fetch_snapshot returns None on network error."""
        mock_urlopen.side_effect = Exception("Connection refused")

        result = camera.fetch_snapshot()

        assert result is None

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_fetch_snapshot_returns_none_on_invalid_data(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test that fetch_snapshot returns None for invalid image data."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not an image"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = camera.fetch_snapshot()

        assert result is None

    @patch.object(Camera, "fetch_snapshot")
    def test_test_connection_success(
        self,
        mock_fetch: MagicMock,
        camera: Camera,
    ) -> None:
        """Test successful connection test."""
        mock_fetch.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        success, dimensions = camera.test_connection()

        assert success is True
        assert dimensions == (640, 480)

    @patch.object(Camera, "fetch_snapshot")
    def test_test_connection_failure(
        self,
        mock_fetch: MagicMock,
        camera: Camera,
    ) -> None:
        """Test failed connection test."""
        mock_fetch.return_value = None

        success, dimensions = camera.test_connection()

        assert success is False
        assert dimensions is None
