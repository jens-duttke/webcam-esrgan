"""Real-ESRGAN image enhancement functionality."""

from __future__ import annotations

import ssl
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from realesrgan import RealESRGANer


def _download_file(url: str, dest: Path) -> None:
    """
    Downloads a file with proper SSL certificate handling.

    Uses certifi certificates if available, falls back to system certificates.

    Args:
        url: URL to download from.
        dest: Destination file path.
    """
    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl_context = ssl.create_default_context()

    with (
        urllib.request.urlopen(url, context=ssl_context) as response,
        open(dest, "wb") as out_file,
    ):
        out_file.write(response.read())


def _apply_torchvision_workaround() -> None:
    """
    Applies workaround for basicsr/torchvision incompatibility.

    basicsr uses an old import that no longer exists in newer torchvision versions.
    """
    import torchvision.transforms.functional as F_tv

    if not hasattr(F_tv, "rgb_to_grayscale"):
        F_tv.rgb_to_grayscale = F_tv.to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = F_tv


class Enhancer:
    """Handles AI-based image enhancement using Real-ESRGAN."""

    MODEL_URL = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.5.0/realesr-general-x4v3.pth"
    )
    MODEL_FILENAME = "realesr-general-x4v3.pth"

    def __init__(
        self,
        target_height: int = 1080,
        upscale_factor: int = 3,
        enhancement_blend: float = 0.8,
        weights_dir: Path | None = None,
    ) -> None:
        """
        Initializes the enhancer.

        Args:
            target_height: Output image height in pixels.
            upscale_factor: AI upscale factor (2-4).
            enhancement_blend: Blend ratio (0.0 = original, 1.0 = AI only).
            weights_dir: Directory for model weights. Defaults to ./weights.
        """
        self.target_height = target_height
        self.upscale_factor = upscale_factor
        self.enhancement_blend = enhancement_blend
        self.weights_dir = weights_dir or Path.cwd() / "weights"
        self._upsampler: RealESRGANer | None = None

    @property
    def is_initialized(self) -> bool:
        """Returns True if the model is loaded."""
        return self._upsampler is not None

    def initialize(self) -> bool:
        """
        Initializes the Real-ESRGAN model.

        Downloads the model if not present and sets up the upsampler.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        print("Loading Real-ESRGAN model...")

        try:
            # Apply workaround before importing realesrgan
            _apply_torchvision_workaround()

            import torch
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact

            # Ensure weights directory exists
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.weights_dir / self.MODEL_FILENAME

            # Download model if not present
            if not model_path.exists():
                print("Downloading model (one-time, ~5MB)...")
                _download_file(self.MODEL_URL, model_path)
                print("Model downloaded!")

            # Network architecture for realesr-general-x4v3
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )

            # Auto-detect GPU availability
            if torch.cuda.is_available():
                device = "cuda"
                half = True
                print(f"Using CUDA ({torch.cuda.get_device_name(0)})")
            else:
                device = "cpu"
                half = False
                print("Using CPU")

            # Create upsampler
            self._upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=half,
                device=device,
            )

            print("Real-ESRGAN loaded successfully!\n")
            return True

        except ImportError as e:
            print("\nError: Real-ESRGAN not installed!")
            print("Please install the required packages:")
            print("  pip install realesrgan")
            print("  pip install torch torchvision")
            print(f"\nDetails: {e}")
            return False
        except Exception as e:
            print(f"\nError loading Real-ESRGAN: {e}")
            traceback.print_exc()
            return False

    def enhance(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
        """
        Enhances an image using Real-ESRGAN.

        The enhancement process:
        1. Pre-shrinks the image so upscaling reaches target height
        2. Applies Real-ESRGAN super-resolution
        3. Blends with original to preserve details
        4. Scales to exact target height

        Args:
            frame: Input BGR image.

        Returns:
            Enhanced BGR image, or None if enhancement failed.
        """
        if self._upsampler is None:
            print("  Error: Enhancer not initialized")
            return None

        h, w = frame.shape[:2]
        print(f"Processing {w}x{h} image with Real-ESRGAN...")

        # BGR to RGB (Real-ESRGAN expects RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()

        try:
            # Pre-shrink so upscaling reaches target height
            target_pre_h = self.target_height // self.upscale_factor

            if h > target_pre_h:
                pre_scale = target_pre_h / h
                pre_h = int(h * pre_scale)
                pre_w = int(w * pre_scale)
                img_small = cv2.resize(
                    img_rgb, (pre_w, pre_h), interpolation=cv2.INTER_AREA
                )
                print(f"  Pre-shrunk to {pre_w}x{pre_h}")
            else:
                img_small = img_rgb

            # Apply Real-ESRGAN
            print(f"  Applying Real-ESRGAN ({self.upscale_factor}x)...")
            output, _ = self._upsampler.enhance(img_small, outscale=self.upscale_factor)

            elapsed = time.time() - start_time
            print(f"  Enhancement completed in {elapsed:.1f}s")

            # RGB back to BGR
            output_bgr: NDArray[np.uint8] = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            # Adjust to exact target size
            out_h, out_w = output_bgr.shape[:2]
            if out_h != self.target_height:
                scale = self.target_height / out_h
                final_w = int(out_w * scale)
                output_bgr = cv2.resize(  # type: ignore[assignment]
                    output_bgr,
                    (final_w, self.target_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            # Blend with original to preserve details
            if self.enhancement_blend < 1.0:
                original_resized = cv2.resize(
                    frame,
                    (output_bgr.shape[1], output_bgr.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                output_bgr = cv2.addWeighted(  # type: ignore[assignment]
                    original_resized,
                    1.0 - self.enhancement_blend,
                    output_bgr,
                    self.enhancement_blend,
                    0,
                )
                ai_pct = int(self.enhancement_blend * 100)
                orig_pct = int((1 - self.enhancement_blend) * 100)
                print(f"  Blend: {ai_pct}% AI, {orig_pct}% Original")

            final_h, final_w = output_bgr.shape[:2]
            print(f"  Final image: {final_w}x{final_h}")

            return output_bgr.astype(np.uint8)

        except Exception as e:
            print(f"  Error in Real-ESRGAN: {e}")
            traceback.print_exc()
            # Fallback: Resize only
            scale = self.target_height / h
            new_w = int(w * scale)
            resized = cv2.resize(
                frame, (new_w, self.target_height), interpolation=cv2.INTER_LANCZOS4
            )
            return resized.astype(np.uint8)
