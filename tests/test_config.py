"""Tests for webcam_interval_capture.config module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from webcam_interval_capture.config import (
    CameraConfig,
    Config,
    EnhanceConfig,
    ImageConfig,
    ReferenceConfig,
    SFTPConfig,
)


class TestCameraConfig:
    """Tests for CameraConfig dataclass."""

    def test_zoom_focus_defaults(self) -> None:
        """Test default values for zoom/focus settings."""
        config = CameraConfig(
            ip="192.168.1.100",
            user="admin",
            password="secret",
        )

        assert config.zoom is None
        assert config.focus is None
        assert config.focus_tolerance == 5

    def test_zoom_focus_custom_values(self) -> None:
        """Test custom zoom/focus values."""
        config = CameraConfig(
            ip="192.168.1.100",
            user="admin",
            password="secret",
            zoom=25,
            focus=224,
            focus_tolerance=10,
        )

        assert config.zoom == 25
        assert config.focus == 224
        assert config.focus_tolerance == 10


class TestSFTPConfig:
    """Tests for SFTPConfig dataclass."""

    def test_enabled_when_all_fields_set(self) -> None:
        """Test that enabled returns True when all fields are set."""
        config = SFTPConfig(
            host="ftp.example.com",
            port=22,
            user="user",
            password="pass",
            path="/var/www",
        )

        assert config.enabled is True

    def test_disabled_when_host_missing(self) -> None:
        """Test that enabled returns False when host is missing."""
        config = SFTPConfig(
            host=None,
            user="user",
            password="pass",
            path="/var/www",
        )

        assert config.enabled is False

    def test_disabled_when_password_missing(self) -> None:
        """Test that enabled returns False when password is missing."""
        config = SFTPConfig(
            host="ftp.example.com",
            user="user",
            password=None,
            path="/var/www",
        )

        assert config.enabled is False

    def test_default_port(self) -> None:
        """Test that default port is 22."""
        config = SFTPConfig()
        assert config.port == 22


class TestImageConfig:
    """Tests for ImageConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are sensible."""
        config = ImageConfig()

        assert config.jpeg_quality == 80
        assert config.avif_quality == 65
        assert config.avif_speed == 4
        assert config.avif_subsampling == "4:2:0"
        assert config.output_dir == "images"


class TestEnhanceConfig:
    """Tests for EnhanceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are sensible."""
        config = EnhanceConfig()

        assert config.max_strength == 0.15
        assert config.brightness_threshold == 0.3
        assert config.wavelet == "db4"
        assert config.levels == 3
        assert config.fusion_mode == "weighted"

    def test_custom_values(self) -> None:
        """Test custom values are applied."""
        config = EnhanceConfig(
            max_strength=0.2,
            brightness_threshold=0.5,
            wavelet="haar",
            levels=4,
            fusion_mode="max_energy",
        )

        assert config.max_strength == 0.2
        assert config.brightness_threshold == 0.5
        assert config.wavelet == "haar"
        assert config.levels == 4
        assert config.fusion_mode == "max_energy"


class TestReferenceConfig:
    """Tests for ReferenceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are sensible."""
        config = ReferenceConfig()

        assert config.path is None
        assert config.hour == 12

    def test_custom_values(self) -> None:
        """Test custom values are applied."""
        config = ReferenceConfig(
            path="/path/to/ref.jpg",
            hour=14,
        )

        assert config.path == "/path/to/ref.jpg"
        assert config.hour == 14


class TestConfig:
    """Tests for main Config class."""

    @patch.dict(
        os.environ,
        {
            "CAMERA_IP": "192.168.1.100",
            "CAMERA_USER": "admin",
            "CAMERA_PASSWORD": "secret",
            "CAMERA_CHANNEL": "1",
            "CAPTURE_INTERVAL": "5",
            "TARGET_HEIGHT": "720",
            "SHOW_PREVIEW": "false",
            "RETENTION_DAYS": "14",
            "TIMESTAMP_FORMAT": "%H:%M",
            "JPEG_QUALITY": "90",
            "AVIF_QUALITY": "50",
            "ENHANCE_MAX_STRENGTH": "0.2",
            "ENHANCE_WAVELET": "haar",
            "REFERENCE_HOUR": "14",
        },
        clear=True,
    )
    def test_from_env_loads_all_settings(self) -> None:
        """Test that from_env correctly loads all environment variables."""
        config = Config.from_env()

        # Camera settings
        assert config.camera.ip == "192.168.1.100"
        assert config.camera.user == "admin"
        assert config.camera.password == "secret"
        assert config.camera.channel == "1"

        # Capture settings
        assert config.capture_interval == 5
        assert config.target_height == 720
        assert config.show_preview is False
        assert config.retention_days == 14
        assert config.timestamp_format == "%H:%M"

        # Image settings
        assert config.image.jpeg_quality == 90
        assert config.image.avif_quality == 50

        # Enhance settings
        assert config.enhance.max_strength == 0.2
        assert config.enhance.wavelet == "haar"

        # Reference settings
        assert config.reference.hour == 14

    @patch.dict(
        os.environ,
        {
            "CAMERA_IP": "192.168.1.100",
            "CAMERA_USER": "admin",
            "CAMERA_PASSWORD": "secret",
        },
        clear=True,
    )
    def test_from_env_uses_defaults(self) -> None:
        """Test that from_env uses defaults for missing optional settings."""
        config = Config.from_env()

        # Check defaults
        assert config.capture_interval == 1
        assert config.target_height == 1080
        assert config.show_preview is True
        assert config.retention_days == 7

        # Enhance defaults
        assert config.enhance.max_strength == 0.15
        assert config.enhance.wavelet == "db4"
        assert config.enhance.levels == 3

        # Reference defaults
        assert config.reference.path is None
        assert config.reference.hour == 12

    @patch.dict(os.environ, {}, clear=True)
    @patch("webcam_interval_capture.config.load_dotenv")
    def test_from_env_exits_without_camera_credentials(
        self, _mock_load_dotenv: MagicMock
    ) -> None:
        """Test that from_env exits when camera credentials are missing."""
        with pytest.raises(SystemExit):
            Config.from_env()

    @patch.dict(
        os.environ,
        {
            "CAMERA_IP": "192.168.1.100",
            "CAMERA_USER": "admin",
            "CAMERA_PASSWORD": "secret",
            "SFTP_HOST": "ftp.example.com",
            "SFTP_USER": "sftp_user",
            "SFTP_PASSWORD": "sftp_pass",
            "SFTP_PATH": "/httpdocs/webcam",
        },
        clear=True,
    )
    def test_sftp_enabled_when_configured(self) -> None:
        """Test that SFTP is enabled when all settings are provided."""
        config = Config.from_env()

        assert config.sftp.enabled is True
        assert config.sftp.host == "ftp.example.com"
        assert config.sftp.path == "/httpdocs/webcam"

    @patch.dict(
        os.environ,
        {
            "CAMERA_IP": "192.168.1.100",
            "CAMERA_USER": "admin",
            "CAMERA_PASSWORD": "secret",
            "CAMERA_ZOOM": "25",
            "CAMERA_FOCUS": "224",
            "CAMERA_FOCUS_TOLERANCE": "10",
        },
        clear=True,
    )
    def test_zoom_focus_from_env(self) -> None:
        """Test that zoom/focus settings are loaded from environment."""
        config = Config.from_env()

        assert config.camera.zoom == 25
        assert config.camera.focus == 224
        assert config.camera.focus_tolerance == 10

    @patch.dict(
        os.environ,
        {
            "CAMERA_IP": "192.168.1.100",
            "CAMERA_USER": "admin",
            "CAMERA_PASSWORD": "secret",
            "REFERENCE_PATH": "/path/to/reference.jpg",
            "REFERENCE_HOUR": "10",
        },
        clear=True,
    )
    def test_reference_settings_from_env(self) -> None:
        """Test that reference settings are loaded from environment."""
        config = Config.from_env()

        assert config.reference.path == "/path/to/reference.jpg"
        assert config.reference.hour == 10
