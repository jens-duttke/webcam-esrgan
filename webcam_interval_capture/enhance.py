"""DWT-based detail transfer image enhancement."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pywt


def compute_auto_strength(image: np.ndarray, max_strength: float = 0.15) -> float:
    """
    Compute adaptive strength based on image brightness.

    - Dark images (night) -> higher strength (up to max_strength)
    - Bright images (day) -> strength approaches 0

    Uses median brightness to be robust against overexposed areas.

    Args:
        image: BGR or grayscale image.
        max_strength: Maximum strength for completely dark images.

    Returns:
        Computed strength value (0 to max_strength).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    median_brightness = float(np.median(gray)) / 255.0  # type: ignore[arg-type]
    strength = max_strength * (1.0 - median_brightness)

    return strength


def dwt_decompose(img: np.ndarray, wavelet: str = "db4", levels: int = 3) -> list:
    """
    Multi-level 2D DWT decomposition.

    Args:
        img: Grayscale image (float32).
        wavelet: Wavelet type ('db4', 'haar', 'sym4', etc.).
        levels: Number of decomposition levels.

    Returns:
        List of coefficients [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)].
    """
    coeffs = pywt.wavedec2(img, wavelet, level=levels)
    return coeffs  # type: ignore[no-any-return]


def dwt_reconstruct(coeffs: list, wavelet: str = "db4") -> np.ndarray:
    """Reconstruct image from DWT coefficients."""
    return pywt.waverec2(coeffs, wavelet)  # type: ignore[no-any-return]


def compute_local_energy(coeff: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Compute local energy of wavelet coefficients.

    Higher energy = more important detail.
    """
    energy = coeff**2
    kernel = np.ones((window_size, window_size), np.float32) / (window_size**2)
    local_energy = cv2.filter2D(energy.astype(np.float32), -1, kernel)
    return local_energy


def fuse_coefficients_max_energy(
    coeff_day: np.ndarray,
    coeff_night: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Fuse detail coefficients using maximum local energy rule.

    This is a proven fusion strategy that preserves the strongest edges.
    """
    energy_day = compute_local_energy(coeff_day, window_size)
    energy_night = compute_local_energy(coeff_night, window_size)

    mask = (energy_day > energy_night).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)  # type: ignore[assignment]

    fused = coeff_day * mask + coeff_night * (1 - mask)
    return fused


def fuse_coefficients_weighted(
    coeff_day: np.ndarray,
    coeff_night: np.ndarray,
    brightness_weight: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Fuse detail coefficients with brightness-weighted blending.

    Dark areas get more day details, bright areas keep night details.
    """
    if brightness_weight.shape != coeff_day.shape:
        weight = cv2.resize(
            brightness_weight,
            (coeff_day.shape[1], coeff_day.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        weight = brightness_weight

    effective_weight = weight * strength
    fused = coeff_day * effective_weight + coeff_night * (1 - effective_weight)
    return fused  # type: ignore[no-any-return]


def dwt_fusion(
    day_img: np.ndarray,
    night_img: np.ndarray,
    brightness_mask: np.ndarray,
    wavelet: str = "db4",
    levels: int = 3,
    strength: float = 0.5,
    fusion_mode: str = "weighted",
) -> np.ndarray:
    """
    DWT-based image fusion.

    Args:
        day_img: Daytime reference (grayscale float32).
        night_img: Nighttime target (grayscale float32).
        brightness_mask: Weight mask (1.0 = use day, 0.0 = use night).
        wavelet: Wavelet type.
        levels: Decomposition levels.
        strength: Overall fusion strength (0-1).
        fusion_mode: 'weighted' or 'max_energy'.

    Returns:
        Fused image.
    """
    day_coeffs = dwt_decompose(day_img, wavelet, levels)
    night_coeffs = dwt_decompose(night_img, wavelet, levels)

    fused_coeffs = []

    # Approximation (LL): ALWAYS use night to preserve illumination
    fused_coeffs.append(night_coeffs[0])

    # Detail coefficients (LH, HL, HH) at each level
    for level in range(1, len(day_coeffs)):
        day_details = day_coeffs[level]
        night_details = night_coeffs[level]

        fused_details = []
        for d_day, d_night in zip(day_details, night_details, strict=True):
            if fusion_mode == "max_energy":
                fused = fuse_coefficients_max_energy(d_day, d_night)
            else:
                fused = fuse_coefficients_weighted(
                    d_day, d_night, brightness_mask, strength
                )
            fused_details.append(fused)

        fused_coeffs.append(tuple(fused_details))

    result = dwt_reconstruct(fused_coeffs, wavelet)

    if result.shape != night_img.shape:
        result = result[: night_img.shape[0], : night_img.shape[1]]

    return result


def to_lab(
    img_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert BGR to LAB, return L, a, b as float32."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def from_lab(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convert LAB channels back to BGR."""
    L_clipped = np.clip(L, 0, 255).astype(np.uint8)
    a_clipped = np.clip(a, 0, 255).astype(np.uint8)
    b_clipped = np.clip(b, 0, 255).astype(np.uint8)
    lab = cv2.merge([L_clipped, a_clipped, b_clipped])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class Enhancer:
    """
    DWT-based detail transfer from day reference to night target.

    Uses proven wavelet-based image fusion algorithms to transfer
    high-frequency details while preserving nighttime atmosphere.
    """

    def __init__(
        self,
        max_strength: float = 0.15,
        brightness_threshold: float = 0.3,
        wavelet: str = "db4",
        levels: int = 3,
        fusion_mode: str = "weighted",
    ) -> None:
        """
        Initialize the enhancer.

        Args:
            max_strength: Maximum auto-strength for dark images (0-1).
            brightness_threshold: Only transfer in areas darker than this (0-1).
            wavelet: Wavelet type ('db4', 'haar', 'sym4', 'bior1.3').
            levels: Number of DWT decomposition levels (2-4 recommended).
            fusion_mode: 'weighted' (brightness-based) or 'max_energy' (edge-based).
        """
        self.max_strength = max_strength
        self.brightness_threshold = brightness_threshold
        self.wavelet = wavelet
        self.levels = levels
        self.fusion_mode = fusion_mode

        print(f"Enhancer: wavelet={wavelet}, levels={levels}, mode={fusion_mode}")

    def compute_brightness_mask(self, night_L: np.ndarray) -> np.ndarray:
        """
        Compute brightness-based transfer mask.

        Dark areas -> high weight (transfer day details)
        Bright areas -> low weight (keep night details)
        """
        normalized = night_L / 255.0
        mask = 1.0 - normalized

        if self.brightness_threshold > 0:
            mask = np.clip(
                (mask - self.brightness_threshold) / (1.0 - self.brightness_threshold),
                0,
                1,
            )

        mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)  # type: ignore[assignment]
        return mask

    def enhance(
        self,
        frame: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        Enhance an image using detail transfer from reference.

        Returns the enhanced image at original resolution.
        If no reference is provided, returns the original frame unchanged.

        Args:
            frame: Input BGR image (nighttime target).
            reference: Optional BGR reference image (daytime).

        Returns:
            Enhanced BGR image at original resolution, or None if enhancement failed.
        """
        h, w = frame.shape[:2]
        start_time = time.time()

        # No reference: return original
        if reference is None:
            print(f"  No reference image, returning original ({w}x{h})")
            return frame.astype(np.uint8)

        # Compute auto-strength based on image brightness
        strength = compute_auto_strength(frame, self.max_strength)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        median_brightness = float(np.median(gray)) / 255.0  # type: ignore[arg-type]
        print(f"  Auto-strength: {strength:.3f} (brightness: {median_brightness:.2f})")

        # Skip if strength is effectively zero
        if strength < 0.001:
            print(
                f"  Skipped enhancement (strength too low), returning original ({w}x{h})"
            )
            return frame.astype(np.uint8)

        # Ensure reference matches frame dimensions
        if reference.shape[:2] != frame.shape[:2]:
            reference = cv2.resize(
                reference,
                (w, h),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Convert to LAB
        day_L, _, _ = to_lab(reference)
        night_L, night_a, night_b = to_lab(frame)

        # Compute brightness mask
        brightness_mask = self.compute_brightness_mask(night_L)

        # Apply DWT fusion on luminance channel
        print(
            f"  DWT fusion ({self.wavelet}, {self.levels} levels, {self.fusion_mode})..."
        )
        fused_L = dwt_fusion(
            day_L,
            night_L,
            brightness_mask,
            wavelet=self.wavelet,
            levels=self.levels,
            strength=strength,
            fusion_mode=self.fusion_mode,
        )

        # Recombine with night chrominance
        result = from_lab(fused_L, night_a, night_b)

        elapsed = time.time() - start_time
        print(f"  Enhanced ({w}x{h}) in {elapsed:.2f}s")

        return result.astype(np.uint8)
