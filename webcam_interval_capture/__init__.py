"""
Webcam Interval Capture

DWT-based image enhancement for IP camera snapshots.
"""

from webcam_interval_capture.camera import Camera
from webcam_interval_capture.config import Config
from webcam_interval_capture.enhance import Enhancer
from webcam_interval_capture.image import add_timestamp, save_images
from webcam_interval_capture.sftp import SFTPUploader

__version__ = "2.0.0"
__all__ = [
    "Config",
    "Camera",
    "Enhancer",
    "SFTPUploader",
    "add_timestamp",
    "save_images",
]
