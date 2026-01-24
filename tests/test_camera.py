"""Tests for webcam_esrgan.camera module."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from webcam_esrgan.camera import Camera, ZoomFocusState
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

    def test_snapshot_url_constructed_correctly(self, camera: Camera) -> None:
        """Test that snapshot URL is correctly constructed."""
        expected = (
            "https://192.168.1.100/cgi-bin/api.cgi"
            "?user=admin&password=secret&cmd=Snap&channel=0"
        )
        assert camera.snapshot_url == expected

    def test_api_url_constructed_correctly(self, camera: Camera) -> None:
        """Test that API URL is correctly constructed."""
        expected = "https://192.168.1.100/cgi-bin/api.cgi?user=admin&password=secret"
        assert camera._api_url == expected

    def test_snapshot_url_with_different_channel(self) -> None:
        """Test snapshot URL with non-default channel."""
        config = CameraConfig(
            ip="10.0.0.50",
            user="webcam",
            password="pass123",
            channel="2",
        )
        camera = Camera(config)

        assert "channel=2" in camera.snapshot_url
        assert "user=webcam" in camera.snapshot_url

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


class TestZoomFocusState:
    """Tests for ZoomFocusState dataclass."""

    def test_zoom_focus_state_creation(self) -> None:
        """Test creating a ZoomFocusState."""
        state = ZoomFocusState(zoom_pos=25, focus_pos=224)

        assert state.zoom_pos == 25
        assert state.focus_pos == 224


class TestCameraZoomFocus:
    """Tests for Camera zoom/focus functionality."""

    @pytest.fixture
    def camera_config(self) -> CameraConfig:
        """Create a test camera configuration."""
        return CameraConfig(
            ip="192.168.1.100",
            user="admin",
            password="secret",
            channel="0",
            zoom=25,
            focus=224,
            focus_tolerance=5,
        )

    @pytest.fixture
    def camera(self, camera_config: CameraConfig) -> Camera:
        """Create a test camera instance."""
        return Camera(camera_config)

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_api_request_success(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test successful API request."""
        response_data = [{"cmd": "GetZoomFocus", "code": 0, "value": {}}]

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = camera._api_request([{"cmd": "GetZoomFocus"}])

        assert result == response_data

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_api_request_returns_none_on_error(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test that API request returns None on network error."""
        mock_urlopen.side_effect = Exception("Connection refused")

        result = camera._api_request([{"cmd": "GetZoomFocus"}])

        assert result is None

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_get_zoom_focus_success(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test successful GetZoomFocus."""
        response_data = [
            {
                "cmd": "GetZoomFocus",
                "code": 0,
                "value": {
                    "ZoomFocus": {
                        "channel": 0,
                        "focus": {"pos": 224},
                        "zoom": {"pos": 25},
                    }
                },
            }
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = camera.get_zoom_focus()

        assert result is not None
        assert result.zoom_pos == 25
        assert result.focus_pos == 224

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_get_zoom_focus_returns_none_on_api_error(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test that GetZoomFocus returns None on API error."""
        response_data = [
            {
                "cmd": "Unknown",
                "code": 1,
                "error": {"detail": "login failed", "rspCode": -7},
            }
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = camera.get_zoom_focus()

        assert result is None

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_set_zoom_focus_focus_only(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test setting focus only."""
        response_data = [{"cmd": "StartZoomFocus", "code": 0}]

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = camera.set_zoom_focus(focus=224)

        assert result is True
        # Verify the correct command was sent
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        sent_data = json.loads(request.data.decode("utf-8"))
        assert len(sent_data) == 1
        assert sent_data[0]["param"]["ZoomFocus"]["op"] == "FocusPos"
        assert sent_data[0]["param"]["ZoomFocus"]["pos"] == 224

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_set_zoom_focus_both(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test setting both zoom and focus."""
        response_data = [
            {"cmd": "StartZoomFocus", "code": 0},
            {"cmd": "StartZoomFocus", "code": 0},
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = camera.set_zoom_focus(zoom=25, focus=224)

        assert result is True
        # Verify both commands were sent
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        sent_data = json.loads(request.data.decode("utf-8"))
        assert len(sent_data) == 2
        assert sent_data[0]["param"]["ZoomFocus"]["op"] == "FocusPos"
        assert sent_data[1]["param"]["ZoomFocus"]["op"] == "ZoomPos"

    @patch("webcam_esrgan.camera.urllib.request.urlopen")
    def test_set_zoom_focus_returns_false_on_error(
        self,
        mock_urlopen: MagicMock,
        camera: Camera,
    ) -> None:
        """Test that set_zoom_focus returns False on API error."""
        response_data = [
            {
                "cmd": "StartZoomFocus",
                "code": 1,
                "error": {"detail": "ability error", "rspCode": -26},
            }
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = camera.set_zoom_focus(zoom=25)

        assert result is False

    def test_set_zoom_focus_returns_true_when_nothing_to_set(
        self,
        camera: Camera,
    ) -> None:
        """Test that set_zoom_focus returns True when nothing to set."""
        result = camera.set_zoom_focus()

        assert result is True

    @patch.object(Camera, "get_zoom_focus")
    def test_ensure_zoom_focus_already_correct(
        self,
        mock_get: MagicMock,
        camera: Camera,
    ) -> None:
        """Test ensure_zoom_focus when values are already correct."""
        mock_get.return_value = ZoomFocusState(zoom_pos=25, focus_pos=224)

        result = camera.ensure_zoom_focus(target_zoom=25, target_focus=224)

        assert result is True
        mock_get.assert_called_once()

    @patch.object(Camera, "get_zoom_focus")
    def test_ensure_zoom_focus_within_tolerance(
        self,
        mock_get: MagicMock,
        camera: Camera,
    ) -> None:
        """Test ensure_zoom_focus when focus is within tolerance."""
        mock_get.return_value = ZoomFocusState(zoom_pos=25, focus_pos=226)

        result = camera.ensure_zoom_focus(
            target_zoom=25, target_focus=224, focus_tolerance=5
        )

        assert result is True

    @patch("webcam_esrgan.camera.time.sleep")
    @patch.object(Camera, "set_zoom_focus")
    @patch.object(Camera, "get_zoom_focus")
    def test_ensure_zoom_focus_adjusts_and_succeeds(
        self,
        mock_get: MagicMock,
        mock_set: MagicMock,
        mock_sleep: MagicMock,
        camera: Camera,
    ) -> None:
        """Test ensure_zoom_focus when adjustment is needed and succeeds."""
        # First call: wrong values, second call: correct values
        mock_get.side_effect = [
            ZoomFocusState(zoom_pos=20, focus_pos=200),
            ZoomFocusState(zoom_pos=25, focus_pos=224),
        ]
        mock_set.return_value = True

        result = camera.ensure_zoom_focus(target_zoom=25, target_focus=224)

        assert result is True
        mock_set.assert_called_once_with(zoom=25, focus=224)
        mock_sleep.assert_called_once_with(1)

    @patch("webcam_esrgan.camera.time.sleep")
    @patch.object(Camera, "set_zoom_focus")
    @patch.object(Camera, "get_zoom_focus")
    def test_ensure_zoom_focus_timeout(
        self,
        mock_get: MagicMock,
        mock_set: MagicMock,
        mock_sleep: MagicMock,
        camera: Camera,
    ) -> None:
        """Test ensure_zoom_focus when adjustment times out."""
        # Always return wrong values
        mock_get.return_value = ZoomFocusState(zoom_pos=20, focus_pos=200)
        mock_set.return_value = True

        result = camera.ensure_zoom_focus(
            target_zoom=25, target_focus=224, max_retries=3
        )

        assert result is False
        assert mock_sleep.call_count == 3

    def test_ensure_zoom_focus_returns_true_when_nothing_to_check(
        self,
        camera: Camera,
    ) -> None:
        """Test ensure_zoom_focus returns True when nothing to check."""
        result = camera.ensure_zoom_focus(target_zoom=None, target_focus=None)

        assert result is True

    @patch.object(Camera, "get_zoom_focus")
    def test_ensure_zoom_focus_returns_false_when_get_fails(
        self,
        mock_get: MagicMock,
        camera: Camera,
    ) -> None:
        """Test ensure_zoom_focus returns False when get_zoom_focus fails."""
        mock_get.return_value = None

        result = camera.ensure_zoom_focus(target_zoom=25, target_focus=224)

        assert result is False
