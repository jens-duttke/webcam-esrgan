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
from datetime import datetime, timedelta

import cv2

from webcam_esrgan.camera import Camera
from webcam_esrgan.config import Config
from webcam_esrgan.enhance import Enhancer
from webcam_esrgan.image import add_timestamp, save_images
from webcam_esrgan.sftp import SFTPUploader

# Global flag for clean shutdown
shutdown_requested = False


def signal_handler(_signum: int, _frame: object) -> None:
    """Handles Ctrl+C cleanly without Fortran errors."""
    global shutdown_requested
    shutdown_requested = True
    print("\n\nShutting down...")


def wait_for_next_interval(
    interval_minutes: int,
    should_exit: Callable[[], bool],
) -> bool:
    """
    Waits until the next capture interval (aligned to clock).

    Args:
        interval_minutes: Minutes between captures.
        should_exit: Callable that returns True if we should stop waiting.

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
        time.sleep(0.1)
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
    print("=" * 50 + "\n")

    # Initialize components
    camera = Camera(config.camera)
    enhancer = Enhancer(
        target_height=config.target_height,
        upscale_factor=config.upscale_factor,
        enhancement_blend=config.enhancement_blend,
    )
    sftp = SFTPUploader(config.sftp, config.retention_days)

    # Initialize Real-ESRGAN
    if not enhancer.initialize():
        print("\nCould not load Real-ESRGAN. Exiting.")
        return 1

    # Test camera connection
    print("Testing camera connection...")
    success, dimensions = camera.test_connection()
    if not success:
        print("Error: Could not fetch image from camera.")
        return 1

    width, height = dimensions  # type: ignore[misc]
    print(f"Connection successful! Image size: {width}x{height}")

    if config.show_preview:
        print("\nClose the window or press Ctrl+C to exit.\n")
    else:
        print("\nPress Ctrl+C to exit.\n")

    # Create preview window
    window_name = "Real-ESRGAN Enhanced Webcam"
    if config.show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def is_window_closed() -> bool:
        """Checks if the OpenCV window has been closed."""
        if not config.show_preview:
            return False
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def should_exit() -> bool:
        """Checks all exit conditions."""
        if shutdown_requested:
            return True
        if config.show_preview:
            if is_window_closed():
                return True
            cv2.waitKey(1)
        return False

    # Wait for first interval
    print(f"Synchronizing to {config.capture_interval}-minute interval...")
    if not wait_for_next_interval(config.capture_interval, should_exit):
        shutdown_requested = True

    # Main loop
    try:
        while not shutdown_requested:
            # Capture image
            print("Capturing image...")
            frame = camera.fetch_snapshot()

            if frame is not None:
                timestamp_now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"  Image captured at {timestamp_now}")

                # Enhance with Real-ESRGAN
                enhanced = enhancer.enhance(frame)

                if enhanced is not None:
                    # Add timestamp overlay
                    final = add_timestamp(enhanced, config.timestamp_format)

                    # Save images
                    current_jpg, current_avif, timestamped = save_images(
                        final, config.image
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
                        display = cv2.resize(final, (0, 0), fx=0.5, fy=0.5)
                        cv2.imshow(window_name, display)

                    if should_exit():
                        break
            else:
                print("  Capture failed")

            # Wait for next interval
            if not wait_for_next_interval(config.capture_interval, should_exit):
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
