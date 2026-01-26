#!/usr/bin/env python3
"""
Webcam Enhancement with DWT Detail Transfer

Captures snapshots from IP cameras and enhances image quality by
transferring high-frequency details from daytime reference images.
"""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import keyboard
import numpy as np

from webcam_interval_capture.archive import ReferenceManager
from webcam_interval_capture.camera import Camera
from webcam_interval_capture.config import Config
from webcam_interval_capture.enhance import Enhancer
from webcam_interval_capture.image import save_images
from webcam_interval_capture.sftp import SFTPUploader

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Global flag for clean shutdown
shutdown_requested = False

# Preview window constants
PREVIEW_INITIAL_HEIGHT = 600


class PreviewState:
    """Manages preview window state for original/enhanced image comparison."""

    def __init__(self, window_name: str) -> None:
        self.show_original = False
        self.original_frame: NDArray[np.uint8] | None = None
        self.enhanced_frame: NDArray[np.uint8] | None = None
        self.window_name = window_name
        self.aspect_ratio: float | None = None
        self.window_initialized = False
        self.window_visible = False

    def mouse_callback(
        self, event: int, _x: int, _y: int, _flags: int, _param: object
    ) -> None:
        """Handle mouse events for image comparison."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.show_original = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.show_original = False

    def toggle_original(self) -> None:
        """Toggle between original and enhanced view (for keyboard)."""
        self.show_original = not self.show_original

    def get_display_image(self) -> NDArray[np.uint8] | None:
        """Get the appropriate image based on current state."""
        if self.show_original and self.original_frame is not None:
            return self.original_frame
        return self.enhanced_frame

    def initialize_window(self, image: NDArray[np.uint8]) -> None:
        """Create and initialize window based on first image's aspect ratio."""
        img_height, img_width = image.shape[:2]
        self.aspect_ratio = img_width / img_height

        # Calculate initial window size
        initial_width = int(PREVIEW_INITIAL_HEIGHT * self.aspect_ratio)

        # Create window with correct initial size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, initial_width, PREVIEW_INITIAL_HEIGHT)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.window_initialized = True
        self.window_visible = True

    def close_window(self) -> None:
        """Close the preview window without exiting the program."""
        if self.window_visible:
            cv2.destroyWindow(self.window_name)
            self.window_visible = False

    def reopen_window(self) -> None:
        """Reopen the preview window if it was closed."""
        if not self.window_visible and self.window_initialized:
            # Re-create window with same settings
            initial_width = int(PREVIEW_INITIAL_HEIGHT * (self.aspect_ratio or 1.0))
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, initial_width, PREVIEW_INITIAL_HEIGHT)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            self.window_visible = True
            # Immediately show current image
            self.update_display()

    def is_window_open(self) -> bool:
        """Check if the window is currently open and visible."""
        if not self.window_visible:
            return False
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            self.window_visible = False
            return False

    def update_display(self) -> None:
        """Update the preview window with the current image (letterboxed)."""
        display = self.get_display_image()
        if display is None:
            return

        # Initialize window on first display, or skip if window is closed
        if not self.window_initialized:
            self.initialize_window(display)
        elif not self.window_visible:
            return  # Window is closed, don't update

        try:
            # Get current window size
            rect = cv2.getWindowImageRect(self.window_name)
            if rect[2] > 0 and rect[3] > 0:
                window_width, window_height = rect[2], rect[3]
                img_height, img_width = display.shape[:2]
                img_aspect = img_width / img_height
                window_aspect = window_width / window_height

                # Calculate scaled size maintaining aspect ratio
                if window_aspect > img_aspect:
                    # Window is wider - fit to height, add horizontal bars
                    new_height = window_height
                    new_width = int(window_height * img_aspect)
                else:
                    # Window is taller - fit to width, add vertical bars
                    new_width = window_width
                    new_height = int(window_width / img_aspect)

                # Resize image
                resized = cv2.resize(display, (new_width, new_height))

                # Create black canvas and center the image (letterboxing)
                canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
                x_offset = (window_width - new_width) // 2
                y_offset = (window_height - new_height) // 2
                canvas[
                    y_offset : y_offset + new_height, x_offset : x_offset + new_width
                ] = resized

                cv2.imshow(self.window_name, canvas)
        except cv2.error:
            pass


def signal_handler(_signum: int, _frame: object) -> None:
    """Handles Ctrl+C cleanly."""
    global shutdown_requested
    shutdown_requested = True
    print("\n\nShutting down...")


def wait_for_next_interval(
    interval_minutes: int,
    should_exit: Callable[[], bool],
    update_display: Callable[[], None] | None = None,
) -> bool:
    """
    Waits until the next capture interval (aligned to clock).

    Args:
        interval_minutes: Minutes between captures.
        should_exit: Callable that returns True if we should stop waiting.
        update_display: Optional callable to update the preview display.

    Returns:
        True if we should continue, False if exit was requested.
    """
    now = datetime.now()

    # Calculate next aligned time
    # E.g. interval=5: captures at :00, :05, :10, :15...
    current_minute = now.minute
    next_minute = ((current_minute // interval_minutes) + 1) * interval_minutes

    if next_minute >= 60:
        next_time = now.replace(
            minute=next_minute % 60,
            second=0,
            microsecond=0,
        ) + timedelta(hours=next_minute // 60)
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)

    # Safety check
    if next_time <= now:
        next_time += timedelta(minutes=interval_minutes)

    seconds_to_wait = (next_time - now).total_seconds()
    print(
        f"\nWaiting {seconds_to_wait:.1f} seconds until {next_time.strftime('%H:%M')}..."
    )

    wait_start = time.time()
    while time.time() - wait_start < seconds_to_wait:
        if should_exit():
            return False
        # Update display for interactive preview (responds to mouse/keyboard)
        if update_display is not None:
            update_display()
        time.sleep(0.05)  # Shorter sleep for more responsive UI
    return True


def main() -> int:
    """Main application entry point."""
    global shutdown_requested

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configuration
    config = Config.from_env()

    # Print banner
    print("=" * 50)
    print("Webcam Enhancement with DWT Detail Transfer")
    print("=" * 50)
    print(f"Camera: {config.camera.ip} (Channel {config.camera.channel})")
    print(f"Target resolution: {config.target_height}p")
    print(f"Output: JPEG={config.target_height}p, AVIF=original resolution")
    print(
        f"Enhancement: max_strength={config.enhance.max_strength}, "
        f"wavelet={config.enhance.wavelet}"
    )
    if config.reference.path:
        print(f"Reference: {config.reference.path} (fixed)")
    else:
        print(
            f"Reference: auto-select from {config.image.output_dir}/ "
            f"(hour {config.reference.hour}:00)"
        )
    if config.sftp.enabled:
        print(f"SFTP upload: {config.sftp.host}:{config.sftp.path}")
    else:
        print("SFTP upload: disabled")
    if config.camera.zoom is not None or config.camera.focus is not None:
        zoom_str = str(config.camera.zoom) if config.camera.zoom is not None else "-"
        focus_str = (
            f"{config.camera.focus} (±{config.camera.focus_tolerance})"
            if config.camera.focus is not None
            else "-"
        )
        print(f"Zoom/Focus: {zoom_str} / {focus_str}")
    print("=" * 50 + "\n")

    # Initialize components
    camera = Camera(config.camera)
    sftp = SFTPUploader(config.sftp, config.retention_days, config.capture_interval)

    # Initialize enhancer
    enhancer = Enhancer(
        max_strength=config.enhance.max_strength,
        brightness_threshold=config.enhance.brightness_threshold,
        wavelet=config.enhance.wavelet,
        levels=config.enhance.levels,
        fusion_mode=config.enhance.fusion_mode,
    )

    # Initialize reference manager for detail transfer
    reference_manager = ReferenceManager(
        output_dir=Path(config.image.output_dir),
        fixed_reference_path=config.reference.path,
        reference_hour=config.reference.hour,
    )

    # Test camera connection
    print("Testing camera connection...")
    success, dimensions = camera.test_connection()
    if not success:
        print("Error: Could not fetch image from camera.")
        return 1

    width, height = dimensions  # type: ignore[misc]
    print(f"Connection successful! Image size: {width}x{height}")

    if config.show_preview:
        print("\nKeys: 'q' = quit, 'w' = reopen window (in window: Space = toggle)\n")
    else:
        print("\nPress 'q' or Ctrl+C to exit.\n")

    # Create preview window
    window_name = "Webcam Enhancement"
    preview = PreviewState(window_name)

    def check_window_closed() -> None:
        """Check if the window was closed by the user and update state."""
        if not config.show_preview:
            return
        if not preview.window_initialized or not preview.window_visible:
            return
        # Check if window was closed via X button
        if not preview.is_window_open():
            preview.window_visible = False

    def handle_keyboard() -> bool:
        """Handle keyboard input. Returns True if exit requested."""
        # Process OpenCV events and window keyboard input
        if config.show_preview and preview.window_visible:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):  # Space toggles original/enhanced (window only)
                preview.toggle_original()
                preview.update_display()

        # Check terminal keyboard input (q and w)
        if keyboard.is_pressed("q"):
            return True
        if keyboard.is_pressed("w") and not preview.window_visible:
            preview.reopen_window()
            time.sleep(0.2)  # Debounce
        return False

    def should_exit() -> bool:
        """Checks all exit conditions."""
        if shutdown_requested:
            return True
        if config.show_preview:
            check_window_closed()  # Update window state (but don't exit)
            if handle_keyboard():
                return True
        return False

    # Create update function for wait loop
    def update_preview() -> None:
        """Update preview display during wait intervals."""
        if config.show_preview:
            preview.update_display()

    # Wait for first interval
    print(f"Synchronizing to {config.capture_interval}-minute interval...")
    if not wait_for_next_interval(config.capture_interval, should_exit, update_preview):
        shutdown_requested = True

    # Main loop
    try:
        while not shutdown_requested:
            # Ensure zoom/focus are correct before capture (no-op if not configured)
            if not camera.ensure_zoom_focus(
                target_zoom=config.camera.zoom,
                target_focus=config.camera.focus,
                focus_tolerance=config.camera.focus_tolerance,
            ):
                print("  Skipping capture due to zoom/focus adjustment timeout")
                if not wait_for_next_interval(
                    config.capture_interval, should_exit, update_preview
                ):
                    break
                continue

            # Capture image
            print("Capturing image...")
            frame = camera.fetch_snapshot()
            capture_time = datetime.now()  # Record capture time immediately

            if frame is not None:
                timestamp_now = capture_time.strftime("%H:%M:%S.%f")[:-3]
                print(f"  Image captured at {timestamp_now}")

                # Get reference image for detail transfer
                reference = reference_manager.get_reference()

                # Process image (enhance with detail transfer)
                print("Processing image...")
                processed = enhancer.enhance(frame, reference)

                if processed is not None:
                    # Update preview frames atomically (both at once)
                    preview.original_frame = frame
                    preview.enhanced_frame = processed

                    # Save images (JPEG resized with timestamp, AVIF original resolution)
                    current_jpg, current_avif, timestamped = save_images(
                        processed,
                        config.image,
                        config.target_height,
                        config.timestamp_format,
                        capture_time,
                    )

                    # Upload via SFTP
                    if config.sftp.enabled:
                        sftp.sync(
                            [
                                (str(current_jpg), current_jpg.name),
                                (str(current_avif), current_avif.name),
                                (str(timestamped), timestamped.name),
                            ]
                        )

                    # Show preview
                    if config.show_preview:
                        preview.update_display()

                    if should_exit():
                        break
            else:
                print("  Capture failed")

            # Wait for next interval
            if not wait_for_next_interval(
                config.capture_interval, should_exit, update_preview
            ):
                break

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    finally:
        print("\nProgram terminated.")
        if config.show_preview:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
