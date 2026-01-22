"""
Webcam Enhancement with Real-ESRGAN

AI-powered image enhancement for IP camera snapshots.
"""

from webcam_esrgan.camera import Camera
from webcam_esrgan.config import Config
from webcam_esrgan.enhance import Enhancer
from webcam_esrgan.image import add_timestamp, save_images
from webcam_esrgan.sftp import SFTPUploader

__version__ = "1.0.0"
__all__ = [
    "Config",
    "Camera",
    "Enhancer",
    "SFTPUploader",
    "add_timestamp",
    "save_images",
]
