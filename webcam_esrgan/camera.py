"""Camera snapshot capture functionality."""

from __future__ import annotations

import json
import ssl
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from webcam_esrgan.config import CameraConfig


@dataclass
class ZoomFocusState:
    """Current zoom and focus state from the camera."""

    zoom_pos: int
    focus_pos: int


class Camera:
    """Handles snapshot capture from IP cameras."""

    def __init__(self, config: CameraConfig) -> None:
        """
        Initializes the camera connection.

        Args:
            config: Camera configuration with IP, credentials, and channel.
        """
        self.config = config

        # Build API URLs
        self._api_url = (
            f"https://{config.ip}/cgi-bin/api.cgi"
            f"?user={config.user}&password={config.password}"
        )
        self.snapshot_url = f"{self._api_url}&cmd=Snap&channel={config.channel}"

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

    def _api_request(
        self, commands: list[dict[str, Any]], timeout: int = 10
    ) -> list[dict[str, Any]] | None:
        """
        Sends a JSON API request to the camera.

        Args:
            commands: List of command dictionaries to send.
            timeout: Request timeout in seconds.

        Returns:
            List of response dictionaries, or None on error.
        """
        try:
            data = json.dumps(commands).encode("utf-8")
            request = urllib.request.Request(
                self._api_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(
                request,
                context=self._ssl_context,
                timeout=timeout,
            ) as response:
                result: list[dict[str, Any]] = json.loads(
                    response.read().decode("utf-8")
                )
                return result
        except Exception as e:
            print(f"  API request error: {e}")
            return None

    def get_zoom_focus(self) -> ZoomFocusState | None:
        """
        Gets the current zoom and focus position from the camera.

        Returns:
            ZoomFocusState with current positions, or None on error.
        """
        commands = [
            {
                "cmd": "GetZoomFocus",
                "action": 0,
                "param": {"channel": int(self.config.channel)},
            }
        ]
        result = self._api_request(commands)
        if result is None or len(result) == 0:
            return None

        response = result[0]
        if response.get("code") != 0:
            error = response.get("error", {})
            print(f"  GetZoomFocus failed: {error.get('detail', 'Unknown error')}")
            return None

        try:
            zf = response["value"]["ZoomFocus"]
            return ZoomFocusState(
                zoom_pos=zf["zoom"]["pos"],
                focus_pos=zf["focus"]["pos"],
            )
        except (KeyError, TypeError) as e:
            print(f"  Failed to parse ZoomFocus response: {e}")
            return None

    def set_zoom_focus(self, zoom: int | None = None, focus: int | None = None) -> bool:
        """
        Sets the zoom and/or focus position on the camera.

        Args:
            zoom: Target zoom position, or None to skip.
            focus: Target focus position, or None to skip.

        Returns:
            True if the commands were sent successfully.
        """
        if zoom is None and focus is None:
            return True

        commands = []
        channel = int(self.config.channel)

        # Focus must be set first if both are specified
        if focus is not None:
            commands.append(
                {
                    "cmd": "StartZoomFocus",
                    "action": 0,
                    "param": {
                        "ZoomFocus": {
                            "channel": channel,
                            "op": "FocusPos",
                            "pos": focus,
                        }
                    },
                }
            )

        if zoom is not None:
            commands.append(
                {
                    "cmd": "StartZoomFocus",
                    "action": 0,
                    "param": {
                        "ZoomFocus": {
                            "channel": channel,
                            "op": "ZoomPos",
                            "pos": zoom,
                        }
                    },
                }
            )

        result = self._api_request(commands)
        if result is None:
            return False

        # Check all responses for errors
        for response in result:
            if response.get("code") != 0:
                error = response.get("error", {})
                print(
                    f"  StartZoomFocus failed: {error.get('detail', 'Unknown error')}"
                )
                return False

        return True

    def ensure_zoom_focus(
        self,
        target_zoom: int | None,
        target_focus: int | None,
        focus_tolerance: int = 5,
        max_retries: int = 30,
    ) -> bool:
        """
        Ensures the camera zoom and focus are set to the expected values.

        Checks the current zoom/focus state and adjusts if necessary.
        Waits up to max_retries seconds for the camera to reach the target.

        Args:
            target_zoom: Expected zoom position, or None to skip zoom check.
            target_focus: Expected focus position, or None to skip focus check.
            focus_tolerance: Acceptable deviation for focus position.
            max_retries: Maximum seconds to wait for adjustment.

        Returns:
            True if zoom/focus are correct, False if timeout or error.
        """
        # If nothing to check, return immediately
        if target_zoom is None and target_focus is None:
            return True

        # Get current state
        current = self.get_zoom_focus()
        if current is None:
            print("  Could not get current zoom/focus state")
            return False

        # Check if values are already correct
        zoom_ok = target_zoom is None or current.zoom_pos == target_zoom
        focus_ok = target_focus is None or (
            target_focus - focus_tolerance
            <= current.focus_pos
            <= target_focus + focus_tolerance
        )

        if zoom_ok and focus_ok:
            return True

        # Determine what needs adjustment
        need_zoom = not zoom_ok
        need_focus = not focus_ok

        if need_zoom and need_focus:
            print(
                f"  Adjusting zoom ({current.zoom_pos} -> {target_zoom}) "
                f"and focus ({current.focus_pos} -> {target_focus})"
            )
        elif need_zoom:
            print(f"  Adjusting zoom ({current.zoom_pos} -> {target_zoom})")
        else:
            print(f"  Adjusting focus ({current.focus_pos} -> {target_focus})")

        # Send adjustment commands
        zoom_to_set = target_zoom if need_zoom else None
        focus_to_set = target_focus if need_focus else None

        if not self.set_zoom_focus(zoom=zoom_to_set, focus=focus_to_set):
            print("  Failed to send zoom/focus adjustment")
            return False

        # Wait for camera to reach target
        for attempt in range(max_retries):
            time.sleep(1)

            current = self.get_zoom_focus()
            if current is None:
                continue

            # Re-check with updated current values
            zoom_ok = target_zoom is None or current.zoom_pos == target_zoom
            focus_ok = target_focus is None or (
                target_focus - focus_tolerance
                <= current.focus_pos
                <= target_focus + focus_tolerance
            )

            if zoom_ok and focus_ok:
                print(
                    f"  Zoom/focus adjusted successfully "
                    f"(zoom={current.zoom_pos}, focus={current.focus_pos})"
                )
                return True

            if (attempt + 1) % 5 == 0:
                print(
                    f"  Still waiting... "
                    f"(zoom={current.zoom_pos}, focus={current.focus_pos})"
                )

        print(f"  Timeout waiting for zoom/focus adjustment after {max_retries}s")
        return False
