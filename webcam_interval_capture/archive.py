"""Reference image management for detail transfer."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np


class ReferenceManager:
    """
    Manages daytime reference images for detail transfer.

    The reference image can be:
    1. A fixed file path (always uses the same reference)
    2. Auto-selected from previous day's noon capture
    """

    def __init__(
        self,
        archive_dir: Path | str,
        fixed_reference_path: Path | str | None = None,
        reference_hour: int = 12,
    ) -> None:
        """
        Initialize the reference manager.

        Args:
            archive_dir: Directory where archived originals are stored.
            fixed_reference_path: Optional fixed reference image path.
                If provided, always uses this image instead of auto-selecting.
            reference_hour: Hour of day to select reference from (0-23).
                Default is 12 (noon) for best daylight.
        """
        self.archive_dir = Path(archive_dir)
        self.fixed_reference_path = (
            Path(fixed_reference_path) if fixed_reference_path else None
        )
        self.reference_hour = reference_hour
        self._cached_reference: np.ndarray | None = None
        self._cached_reference_path: Path | None = None

    def get_reference(self) -> np.ndarray | None:
        """
        Get the current reference image.

        Returns:
            BGR reference image, or None if not available.
        """
        # Fixed reference path
        if self.fixed_reference_path is not None:
            return self._load_reference(self.fixed_reference_path)

        # Auto-select from yesterday's archive
        yesterday = datetime.now() - timedelta(days=1)
        reference_path = self._find_best_reference(yesterday)

        if reference_path is None:
            return None

        return self._load_reference(reference_path)

    def _load_reference(self, path: Path) -> np.ndarray | None:
        """Load and cache a reference image."""
        if not path.exists():
            print(f"  Reference not found: {path}")
            return None

        # Return cached if same path
        if self._cached_reference_path == path and self._cached_reference is not None:
            return self._cached_reference

        # Load new reference
        img = cv2.imread(str(path))
        if img is None:
            print(f"  Could not load reference: {path}")
            return None

        self._cached_reference = img
        self._cached_reference_path = path
        print(f"  Using reference: {path.name}")
        return img

    def _find_best_reference(self, date: datetime) -> Path | None:
        """
        Find the best reference image for a given date.

        Looks for an image captured around the reference hour.
        """
        if not self.archive_dir.exists():
            return None

        # Pattern: original_YYYY-MM-DD-HH-MM.jpg (or .png)
        date_prefix = date.strftime("%Y-%m-%d")
        target_hour = self.reference_hour

        best_match: Path | None = None
        best_hour_diff = 24

        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for path in self.archive_dir.glob(f"original_{date_prefix}*{ext[1:]}"):
                # Extract hour from filename
                try:
                    # original_2024-01-15-12-30.jpg -> extract hour (12)
                    name = path.stem  # original_2024-01-15-12-30
                    parts = name.split("-")
                    if len(parts) >= 5:
                        hour = int(parts[3])
                        hour_diff = abs(hour - target_hour)
                        if hour_diff < best_hour_diff:
                            best_hour_diff = hour_diff
                            best_match = path
                except (ValueError, IndexError):
                    continue

        return best_match
