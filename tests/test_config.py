"""Tests for webcam_esrgan.config module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from webcam_esrgan.config import CameraConfig, Config, ImageConfig, SFTPConfig


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

    def test_disabled_when_user_missing(self) -> None:
        """Test that enabled returns False when user is missing."""
        config = SFTPConfig(
            host="ftp.example.com",
            user=None,
            password="pass",
            path="/var/www",
        )

        assert config.enabled is False

    def test_disabled_when_path_missing(self) -> None:
        """Test that enabled returns False when path is missing."""
        config = SFTPConfig(
            host="ftp.example.com",
            user="user",
            password="pass",
            path=None,
        )

        assert config.enabled is False

    def test_default_port(self) -> None:
        """Test that default port is 22."""
        config = SFTPConfig()
        assert config.port == 22

    def test_custom_port(self) -> None:
        """Test that custom port is used."""
        config = SFTPConfig(port=2222)
        assert config.port == 2222


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
            "UPSCALE_FACTOR": "2",
            "ENHANCEMENT_BLEND": "0.6",
            "SHOW_PREVIEW": "false",
            "RETENTION_DAYS": "14",
            "TIMESTAMP_FORMAT": "%H:%M",
            "JPEG_QUALITY": "90",
            "AVIF_QUALITY": "50",
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
        assert config.upscale_factor == 2
        assert config.enhancement_blend == 0.6
        assert config.show_preview is False
        assert config.retention_days == 14
        assert config.timestamp_format == "%H:%M"

        # Image settings
        assert config.image.jpeg_quality == 90
        assert config.image.avif_quality == 50

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
        assert config.upscale_factor == 2
        assert config.enhancement_blend == 0.8
        assert config.show_preview is True
        assert config.retention_days == 7
        assert config.tile_size == 400
        assert config.max_downscale_factor == 2

    @patch.dict(os.environ, {}, clear=True)
    @patch("webcam_esrgan.config.load_dotenv")
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
        },
        clear=True,
    )
    def test_zoom_focus_defaults_when_not_set(self) -> None:
        """Test that zoom/focus have correct defaults when not set."""
        config = Config.from_env()

        assert config.camera.zoom is None
        assert config.camera.focus is None
        assert config.camera.focus_tolerance == 5
