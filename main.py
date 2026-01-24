#!/usr/bin/env python3
"""
Webcam Enhancement with Real-ESRGAN

Captures snapshots from IP cameras and enhances image quality
using Real-ESRGAN AI upscaling.
"""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import cv2
import numpy as np

from webcam_esrgan.camera import Camera
from webcam_esrgan.config import Config
from webcam_esrgan.enhance import Enhancer
from webcam_esrgan.image import add_timestamp, save_images
from webcam_esrgan.sftp import SFTPUploader

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
    """Handles Ctrl+C cleanly without Fortran errors."""
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
    print("Webcam Enhancement with Real-ESRGAN")
    print("=" * 50)
    print(f"Camera: {config.camera.ip} (Channel {config.camera.channel})")
    print(f"Target resolution: {config.target_height}p")
    print(
        f"Upscale factor: {config.upscale_factor}x "
        f"(Pre-shrink to {config.target_height // config.upscale_factor}p)"
    )
    print(
        f"AI blend: {int(config.enhancement_blend * 100)}% AI / "
        f"{int((1 - config.enhancement_blend) * 100)}% Original"
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

    # Initialize Real-ESRGAN (skip if enhancement_blend is 0)
    enhancer: Enhancer | None = None
    if config.enhancement_blend > 0:
        enhancer = Enhancer(
            target_height=config.target_height,
            upscale_factor=config.upscale_factor,
            enhancement_blend=config.enhancement_blend,
            tile_size=config.tile_size,
            max_downscale_factor=config.max_downscale_factor,
        )
        if not enhancer.initialize():
            print("\nCould not load Real-ESRGAN. Exiting.")
            return 1
    else:
        print("AI enhancement disabled (ENHANCEMENT_BLEND=0), skipping model load.")

    # Test camera connection
    print("Testing camera connection...")
    success, dimensions = camera.test_connection()
    if not success:
        print("Error: Could not fetch image from camera.")
        return 1

    width, height = dimensions  # type: ignore[misc]
    print(f"Connection successful! Image size: {width}x{height}")

    if config.show_preview:
        print("\nPress 'q' or Ctrl+C to exit. Close window with X, reopen with 'w'.\n")
    else:
        print("\nPress Ctrl+C to exit.\n")

    # Create preview window
    window_name = "Real-ESRGAN Enhanced Webcam"
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
        if not config.show_preview:
            return False

        # Always process keyboard, even when window is closed (to allow reopening)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            return True
        if key == ord(" "):  # Spacebar toggles original/enhanced
            preview.toggle_original()
            preview.update_display()
        if (
            key == ord("w")
            and not preview.window_visible
            and preview.window_initialized
        ):  # 'w' reopens window if closed
            preview.reopen_window()
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

                # Process image (enhance with AI or just resize)
                if enhancer is not None:
                    # Enhance with Real-ESRGAN in background thread
                    # This keeps the UI responsive during processing
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future: Future[NDArray[np.uint8] | None] = executor.submit(
                            enhancer.enhance, frame
                        )

                        # Keep UI responsive while processing
                        while not future.done():
                            if config.show_preview:
                                check_window_closed()
                                preview.update_display()
                                key = cv2.waitKey(50) & 0xFF
                                if key == ord("q") or key == 27:
                                    shutdown_requested = True
                                elif (
                                    key == ord("w")
                                    and not preview.window_visible
                                    and preview.window_initialized
                                ):
                                    preview.reopen_window()
                            else:
                                time.sleep(0.05)

                            if shutdown_requested:
                                # Can't cancel PyTorch, but we can exit after it finishes
                                print("\n  Finishing current operation...")
                                break

                        processed = future.result()
                else:
                    # No enhancement - just resize to target height
                    h, w = frame.shape[:2]
                    scale = config.target_height / h
                    new_w = int(w * scale)
                    processed = cv2.resize(
                        frame,
                        (new_w, config.target_height),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    print(f"  Resized to {new_w}x{config.target_height} (no AI)")

                if processed is not None:
                    # Add timestamp overlay for JPEG only (use capture time)
                    with_timestamp = add_timestamp(
                        processed, config.timestamp_format, capture_time
                    )

                    # Update preview frames atomically (both at once)
                    preview.original_frame = frame
                    preview.enhanced_frame = with_timestamp

                    # Save images (JPEG with timestamp, AVIF without)
                    current_jpg, current_avif, timestamped = save_images(
                        with_timestamp, processed, config.image, capture_time
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
