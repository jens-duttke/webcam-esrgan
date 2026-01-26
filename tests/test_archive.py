"""Tests for webcam_interval_capture.archive module."""

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from webcam_interval_capture.archive import ReferenceManager


class TestReferenceManager:
    """Tests for ReferenceManager class."""

    @pytest.fixture
    def archive_dir(self, tmp_path: Path) -> Path:
        """Create a temporary archive directory."""
        archive = tmp_path / "archive"
        archive.mkdir()
        return archive

    @pytest.fixture
    def test_reference(self, archive_dir: Path) -> Path:
        """Create a test reference image."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 150
        yesterday = datetime.now() - timedelta(days=1)
        filename = f"original_{yesterday.strftime('%Y-%m-%d')}-12-00.jpg"
        filepath = archive_dir / filename
        cv2.imwrite(str(filepath), img)
        return filepath

    def test_get_reference_with_fixed_path(
        self, archive_dir: Path, test_reference: Path
    ) -> None:
        """Test that fixed reference path is used."""
        manager = ReferenceManager(
            archive_dir=archive_dir, fixed_reference_path=test_reference
        )

        result = manager.get_reference()

        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_get_reference_fixed_path_not_found(self, archive_dir: Path) -> None:
        """Test that missing fixed path returns None."""
        manager = ReferenceManager(
            archive_dir=archive_dir,
            fixed_reference_path=archive_dir / "nonexistent.jpg",
        )

        result = manager.get_reference()

        assert result is None

    def test_get_reference_auto_select(
        self,
        archive_dir: Path,
        test_reference: Path,  # noqa: ARG002
    ) -> None:
        """Test that auto-selection finds yesterday's noon image."""
        manager = ReferenceManager(archive_dir=archive_dir, reference_hour=12)

        result = manager.get_reference()

        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_get_reference_auto_select_no_match(self, archive_dir: Path) -> None:
        """Test that auto-selection returns None when no match."""
        manager = ReferenceManager(archive_dir=archive_dir, reference_hour=12)

        result = manager.get_reference()

        assert result is None

    def test_get_reference_caches_result(
        self, archive_dir: Path, test_reference: Path
    ) -> None:
        """Test that reference is cached."""
        manager = ReferenceManager(
            archive_dir=archive_dir, fixed_reference_path=test_reference
        )

        result1 = manager.get_reference()
        result2 = manager.get_reference()

        assert result1 is result2

    def test_find_best_reference_prefers_target_hour(self, archive_dir: Path) -> None:
        """Test that auto-selection prefers target hour."""
        yesterday = datetime.now() - timedelta(days=1)

        for hour in [10, 12, 14]:
            filename = f"original_{yesterday.strftime('%Y-%m-%d')}-{hour:02d}-00.jpg"
            filepath = archive_dir / filename
            cv2.imwrite(str(filepath), np.ones((10, 10, 3), dtype=np.uint8) * hour)

        manager = ReferenceManager(archive_dir=archive_dir, reference_hour=12)
        result = manager.get_reference()

        assert result is not None
        assert result[0, 0, 0] == 12
