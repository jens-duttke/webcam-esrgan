"""Configuration management for webcam_esrgan."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class CameraConfig:
    """Camera connection settings."""

    ip: str
    user: str
    password: str
    channel: str = "0"

    @property
    def snapshot_url(self) -> str:
        """Constructs the Reolink API snapshot URL."""
        return (
            f"https://{self.ip}/cgi-bin/api.cgi"
            f"?cmd=Snap&channel={self.channel}"
            f"&user={self.user}&password={self.password}"
        )


@dataclass
class SFTPConfig:
    """SFTP upload settings."""

    host: str | None = None
    port: int = 22
    user: str | None = None
    password: str | None = None
    path: str | None = None

    @property
    def enabled(self) -> bool:
        """Returns True if all SFTP settings are configured."""
        return all([self.host, self.user, self.password, self.path])


@dataclass
class ImageConfig:
    """Image quality settings."""

    jpeg_quality: int = 80
    avif_quality: int = 65
    avif_speed: int = 4
    avif_subsampling: str = "4:2:0"
    output_dir: str = "images"


@dataclass
class Config:
    """Main configuration container."""

    camera: CameraConfig
    sftp: SFTPConfig = field(default_factory=SFTPConfig)
    image: ImageConfig = field(default_factory=ImageConfig)

    # Capture settings
    capture_interval: int = 1
    target_height: int = 1080
    upscale_factor: int = 3
    enhancement_blend: float = 0.8
    show_preview: bool = True
    retention_days: int = 7
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> Config:
        """
        Loads configuration from environment variables.

        Args:
            env_path: Optional path to .env file. If None, searches for
                     .env.local or .env in the current directory.

        Returns:
            Configured Config instance.

        Raises:
            SystemExit: If required camera credentials are missing.
        """
        if env_path is None:
            # Look for .env.local first, then .env
            for filename in (".env.local", ".env"):
                path = Path.cwd() / filename
                if path.exists():
                    env_path = path
                    break

        if env_path and env_path.exists():
            load_dotenv(env_path)

        # Validate required camera credentials
        camera_ip = os.getenv("CAMERA_IP")
        camera_user = os.getenv("CAMERA_USER")
        camera_password = os.getenv("CAMERA_PASSWORD")

        if not all([camera_ip, camera_user, camera_password]):
            print("ERROR: Camera credentials not found!")
            print("Please create a .env.local file with the following variables:")
            print("  CAMERA_IP=192.168.x.x")
            print("  CAMERA_USER=your_username")
            print("  CAMERA_PASSWORD=your_password")
            print("  CAMERA_CHANNEL=0")
            print("\nSee .env.example for a template.")
            sys.exit(1)

        return cls(
            camera=CameraConfig(
                ip=camera_ip,  # type: ignore[arg-type]
                user=camera_user,  # type: ignore[arg-type]
                password=camera_password,  # type: ignore[arg-type]
                channel=os.getenv("CAMERA_CHANNEL", "0"),
            ),
            sftp=SFTPConfig(
                host=os.getenv("SFTP_HOST"),
                port=int(os.getenv("SFTP_PORT", "22")),
                user=os.getenv("SFTP_USER"),
                password=os.getenv("SFTP_PASSWORD"),
                path=os.getenv("SFTP_PATH"),
            ),
            image=ImageConfig(
                jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")),
                avif_quality=int(os.getenv("AVIF_QUALITY", "65")),
                avif_speed=int(os.getenv("AVIF_SPEED", "4")),
                avif_subsampling=os.getenv("AVIF_SUBSAMPLING", "4:2:0"),
                output_dir=os.getenv("OUTPUT_DIR", "images"),
            ),
            capture_interval=int(os.getenv("CAPTURE_INTERVAL", "1")),
            target_height=int(os.getenv("TARGET_HEIGHT", "1080")),
            upscale_factor=int(os.getenv("UPSCALE_FACTOR", "3")),
            enhancement_blend=float(os.getenv("ENHANCEMENT_BLEND", "0.8")),
            show_preview=os.getenv("SHOW_PREVIEW", "true").lower() == "true",
            retention_days=int(os.getenv("RETENTION_DAYS", "7")),
            timestamp_format=os.getenv("TIMESTAMP_FORMAT", "%Y-%m-%d %H:%M:%S"),
        )
