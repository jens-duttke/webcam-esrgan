"""Camera snapshot capture functionality."""

from __future__ import annotations

import ssl
import urllib.request

import cv2
import numpy as np
from numpy.typing import NDArray

from webcam_esrgan.config import CameraConfig


class Camera:
    """Handles snapshot capture from IP cameras."""

    def __init__(self, config: CameraConfig) -> None:
        """
        Initializes the camera connection.

        Args:
            config: Camera configuration with IP, credentials, and channel.
        """
        self.config = config
        self.snapshot_url = config.snapshot_url

        # Create SSL context that ignores certificate errors (for local cameras)
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE

    def fetch_snapshot(self, timeout: int = 10) -> NDArray[np.uint8] | None:
        """
        Fetches a single snapshot from the camera.

        Args:
            timeout: Request timeout in seconds.

        Returns:
            BGR image as numpy array, or None if fetch failed.
        """
        try:
            with urllib.request.urlopen(
                self.snapshot_url,
                context=self._ssl_context,
                timeout=timeout,
            ) as response:
                img_bytes = response.read()
                img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    return None
                return frame.astype(np.uint8)
        except Exception as e:
            print(f"  Error fetching image: {e}")
            return None

    def test_connection(self) -> tuple[bool, tuple[int, int] | None]:
        """
        Tests the camera connection.

        Returns:
            Tuple of (success, (width, height)) or (False, None) on failure.
        """
        frame = self.fetch_snapshot()
        if frame is not None:
            h, w = frame.shape[:2]
            return True, (w, h)
        return False, None
