"""Tests for webcam_interval_capture.rsync module."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from webcam_interval_capture.config import RsyncConfig
from webcam_interval_capture.rsync import RsyncUploader


class TestRsyncUploader:
    """Tests for RsyncUploader class."""

    @pytest.fixture
    def rsync_config(self) -> RsyncConfig:
        """Create a test rsync configuration."""
        return RsyncConfig(
            host="stardust.uberspace.de",
            port=22,
            user="isabell",
            ssh_key="/home/user/.ssh/deploy_key",
            path="/var/www/virtual/isabell/html/webcam",
        )

    @pytest.fixture
    def disabled_config(self) -> RsyncConfig:
        """Create a disabled rsync configuration."""
        return RsyncConfig()

    @pytest.fixture
    def uploader(self, rsync_config: RsyncConfig) -> RsyncUploader:
        """Create a test uploader instance."""
        return RsyncUploader(rsync_config, retention_days=7, capture_interval=5)

    def test_sync_returns_false_when_disabled(
        self,
        disabled_config: RsyncConfig,
    ) -> None:
        """Test that sync returns False when rsync is disabled."""
        uploader = RsyncUploader(disabled_config)

        result = uploader.sync([("local.jpg", "remote.jpg")])

        assert result is False

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_sync_uploads_files(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that sync uploads all provided files."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        files = [
            ("/tmp/current.jpg", "webcam_current.jpg"),
            ("/tmp/timestamped.avif", "webcam_2026-01-21-12-00.avif"),
        ]

        result = uploader.sync(files)

        assert result is True
        # 1 batch upload + 1 list + 1 log upload = 3 subprocess calls
        assert mock_run.call_count == 3

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_upload_uses_rsync_command(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that upload calls rsync with correct arguments."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        uploader.sync(
            [
                ("/tmp/test.jpg", "webcam_current.jpg"),
                ("/tmp/test.avif", "webcam_current.avif"),
            ]
        )

        # First call should be rsync batch upload
        rsync_call = mock_run.call_args_list[0]
        cmd = rsync_call[0][0]
        assert cmd[0] == "rsync"
        assert "-az" in cmd
        assert "-e" in cmd
        # Both files in a single call
        assert "/tmp/test.jpg" in cmd
        assert "/tmp/test.avif" in cmd
        assert "isabell@stardust.uberspace.de:" in cmd[-1]

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_list_remote_files_parses_rsync_output(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that rsync --list-only output is parsed correctly."""
        rsync_listing = (
            "drwxr-xr-x        123 2026/01/20 10:00:00 .\n"
            "-rw-r--r--      50000 2026/01/20 10:00:00 webcam_current.jpg\n"
            "-rw-r--r--      60000 2026/01/20 10:00:00 webcam_2026-01-20-10-00.avif\n"
            "-rw-r--r--       1234 2026/01/20 10:00:00 webcam_log.json\n"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=rsync_listing, stderr="")

        files = uploader._list_remote_files("/remote/dir")

        assert "webcam_current.jpg" in files
        assert "webcam_2026-01-20-10-00.avif" in files
        assert "webcam_log.json" in files
        # Directory entries should be skipped
        assert "." not in files

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_list_remote_files_handles_failure(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that listing failure returns empty list."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")

        files = uploader._list_remote_files("/remote/dir")

        assert files == []

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_cleanup_removes_old_files(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that cleanup removes files older than retention_days."""
        old_date = datetime.now() - timedelta(days=10)
        recent_date = datetime.now() - timedelta(days=1)

        old_filename = f"webcam_{old_date.strftime('%Y-%m-%d-%H-%M')}.avif"
        recent_filename = f"webcam_{recent_date.strftime('%Y-%m-%d-%H-%M')}.avif"

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        remote_files = [old_filename, recent_filename, "webcam_current.jpg"]
        deleted = uploader._cleanup_old_files("/remote/dir", remote_files)

        assert old_filename in deleted
        assert recent_filename not in deleted
        assert "webcam_current.jpg" not in deleted

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_cleanup_uses_rsync_delete(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that cleanup uses rsync --delete with include/exclude filters."""
        old_date = datetime.now() - timedelta(days=10)
        old_filename = f"webcam_{old_date.strftime('%Y-%m-%d-%H-%M')}.avif"

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        uploader._cleanup_old_files("/remote/dir", [old_filename])

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "--delete" in cmd
        assert f"--include={old_filename}" in cmd_str
        assert "--exclude=*" in cmd_str

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_cleanup_skips_invalid_date_files(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that cleanup skips files with invalid date format."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        old_date = datetime.now() - timedelta(days=10)
        old_valid = f"webcam_{old_date.strftime('%Y-%m-%d-%H-%M')}.avif"
        remote_files = ["webcam_invalid-date.avif", old_valid]

        deleted = uploader._cleanup_old_files("/remote/dir", remote_files)

        assert old_valid in deleted
        assert "webcam_invalid-date.avif" not in deleted

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_sync_handles_rsync_error(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that sync handles rsync errors gracefully."""
        mock_run.side_effect = Exception("Connection refused")

        result = uploader.sync([("local.jpg", "remote.jpg")])

        assert result is False

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_log_contains_history_images(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that log file contains history images from remote listing."""
        recent = datetime.now() - timedelta(hours=1)
        hist1 = f"webcam_{recent.strftime('%Y-%m-%d-%H-%M')}.avif"
        hist2_time = recent - timedelta(hours=1)
        hist2 = f"webcam_{hist2_time.strftime('%Y-%m-%d-%H-%M')}.avif"

        rsync_listing = (
            f"-rw-r--r--  50000 2026/01/20 10:00:00 {hist1}\n"
            f"-rw-r--r--  60000 2026/01/20 10:00:00 {hist2}\n"
            "-rw-r--r--  70000 2026/01/20 10:00:00 webcam_current.jpg\n"
            "-rw-r--r--  80000 2026/01/20 10:00:00 webcam_current.avif\n"
            "-rw-r--r--   1234 2026/01/20 10:00:00 webcam_log.json\n"
            "-rw-r--r--   5678 2026/01/20 10:00:00 other_file.txt\n"
        )

        def side_effect(cmd: list[str], **_kwargs: object) -> MagicMock:
            result = MagicMock(returncode=0, stderr="")
            if "--list-only" in cmd:
                result.stdout = rsync_listing
            else:
                result.stdout = ""
            return result

        mock_run.side_effect = side_effect

        uploader.sync([])

        # Find the rsync log upload call (the one uploading webcam_log.tmp.json)
        rsync_calls = [
            c for c in mock_run.call_args_list if "webcam_log.json" in str(c[0][0][-1])
        ]
        assert len(rsync_calls) == 1

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_upload_failure_prints_error(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that upload failure prints error message."""

        def side_effect(cmd: list[str], **_kwargs: object) -> MagicMock:
            result = MagicMock(stderr="")
            if "-az" in cmd and "--list-only" not in cmd:
                result.returncode = 1
                result.stderr = "rsync: connection unexpectedly closed"
            else:
                result.returncode = 0
                result.stdout = ""
            return result

        mock_run.side_effect = side_effect

        uploader.sync([("/tmp/test.jpg", "webcam_current.jpg")])

        captured = capsys.readouterr()
        assert "Rsync upload failed" in captured.out

    def test_rsync_config_enabled_requires_all_fields(self) -> None:
        """Test that RsyncConfig.enabled requires all fields."""
        assert not RsyncConfig().enabled
        assert not RsyncConfig(host="h").enabled
        assert not RsyncConfig(host="h", user="u").enabled
        assert not RsyncConfig(host="h", user="u", ssh_key="k").enabled
        assert RsyncConfig(host="h", user="u", ssh_key="k", path="/p").enabled

    @patch("webcam_interval_capture.rsync.subprocess.run")
    def test_ssh_key_in_rsync_command(
        self,
        mock_run: MagicMock,
        uploader: RsyncUploader,
    ) -> None:
        """Test that SSH key is included in rsync -e argument."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        uploader.sync([("/tmp/test.jpg", "webcam_current.jpg")])

        rsync_call = mock_run.call_args_list[0]
        cmd = rsync_call[0][0]
        # Find the -e argument
        e_index = cmd.index("-e")
        ssh_cmd = cmd[e_index + 1]
        assert "/home/user/.ssh/deploy_key" in ssh_cmd
        assert "BatchMode=yes" in ssh_cmd
