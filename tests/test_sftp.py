"""Tests for webcam_esrgan.sftp module."""

import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from webcam_esrgan.config import SFTPConfig
from webcam_esrgan.sftp import SFTPUploader


class TestSFTPUploader:
    """Tests for SFTPUploader class."""

    @pytest.fixture
    def sftp_config(self) -> SFTPConfig:
        """Create a test SFTP configuration."""
        return SFTPConfig(
            host="ftp.example.com",
            port=22,
            user="testuser",
            password="testpass",
            path="/var/www/webcam",
        )

    @pytest.fixture
    def disabled_config(self) -> SFTPConfig:
        """Create a disabled SFTP configuration."""
        return SFTPConfig()

    @pytest.fixture
    def uploader(self, sftp_config: SFTPConfig) -> SFTPUploader:
        """Create a test uploader instance."""
        return SFTPUploader(sftp_config, retention_days=7)

    @pytest.fixture
    def mock_paramiko(self) -> MagicMock:
        """Create a mock paramiko module."""
        mock = MagicMock()
        mock_sftp = MagicMock()
        mock_sftp.listdir_attr.return_value = []
        mock_sftp.listdir.return_value = []
        mock_sftp.file.return_value.__enter__ = MagicMock()
        mock_sftp.file.return_value.__exit__ = MagicMock()

        mock_transport = MagicMock()
        mock.Transport.return_value = mock_transport
        mock.SFTPClient.from_transport.return_value = mock_sftp

        return mock

    def test_sync_returns_false_when_disabled(
        self,
        disabled_config: SFTPConfig,
    ) -> None:
        """Test that sync returns False when SFTP is disabled."""
        uploader = SFTPUploader(disabled_config)

        result = uploader.sync([("local.jpg", "remote.jpg")])

        assert result is False

    def test_sync_uploads_files(
        self,
        uploader: SFTPUploader,
        mock_paramiko: MagicMock,
    ) -> None:
        """Test that sync uploads all provided files."""
        with patch.dict(sys.modules, {"paramiko": mock_paramiko}):
            files = [
                ("/tmp/current.jpg", "webcam_current.jpg"),
                ("/tmp/timestamped.avif", "webcam_2026-01-21-12-00.avif"),
            ]

            result = uploader.sync(files)

            assert result is True
            mock_sftp = mock_paramiko.SFTPClient.from_transport.return_value
            assert mock_sftp.put.call_count == 2

    def test_cleanup_removes_old_files(
        self,
        uploader: SFTPUploader,
        mock_paramiko: MagicMock,
    ) -> None:
        """Test that cleanup removes files older than retention_days."""
        mock_sftp = mock_paramiko.SFTPClient.from_transport.return_value

        # Create file entries - one old, one recent
        old_date = datetime.now() - timedelta(days=10)
        recent_date = datetime.now() - timedelta(days=1)

        old_entry = MagicMock()
        old_entry.filename = f"webcam_{old_date.strftime('%Y-%m-%d-%H-%M')}.avif"

        recent_entry = MagicMock()
        recent_entry.filename = f"webcam_{recent_date.strftime('%Y-%m-%d-%H-%M')}.avif"

        current_entry = MagicMock()
        current_entry.filename = "webcam_current.jpg"

        mock_sftp.listdir_attr.return_value = [old_entry, recent_entry, current_entry]
        mock_sftp.listdir.return_value = [recent_entry.filename]

        with patch.dict(sys.modules, {"paramiko": mock_paramiko}):
            uploader.sync([])

            # Should remove only the old file
            assert mock_sftp.remove.call_count == 1
            remove_call_arg = mock_sftp.remove.call_args[0][0]
            assert old_entry.filename in remove_call_arg

    def test_sync_handles_connection_error(
        self,
        uploader: SFTPUploader,
        mock_paramiko: MagicMock,
    ) -> None:
        """Test that sync handles connection errors gracefully."""
        mock_paramiko.Transport.return_value.connect.side_effect = Exception(
            "Connection refused"
        )

        with patch.dict(sys.modules, {"paramiko": mock_paramiko}):
            result = uploader.sync([("local.jpg", "remote.jpg")])

            assert result is False

    def test_log_contains_all_images(
        self,
        uploader: SFTPUploader,
        mock_paramiko: MagicMock,
    ) -> None:
        """Test that log file contains all image files."""
        mock_sftp = mock_paramiko.SFTPClient.from_transport.return_value
        mock_sftp.listdir_attr.return_value = []
        mock_sftp.listdir.return_value = [
            "webcam_2026-01-20-10-00.avif",
            "webcam_2026-01-20-11-00.avif",
            "webcam_2026-01-20-12-00.avif",
            "webcam_current.jpg",
            "webcam_log.json",
            "other_file.txt",
        ]

        mock_file = MagicMock()
        mock_sftp.file.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_sftp.file.return_value.__exit__ = MagicMock()

        with patch.dict(sys.modules, {"paramiko": mock_paramiko}):
            uploader.sync([])

            # Check that file was written with correct content
            write_call = mock_file.write.call_args[0][0]
            assert "webcam_2026-01-20-10-00.avif" in write_call
            assert "webcam_2026-01-20-11-00.avif" in write_call
            assert "webcam_2026-01-20-12-00.avif" in write_call
            assert "webcam_current.jpg" not in write_call
            assert "webcam_current.avif" not in write_call
            assert "other_file.txt" not in write_call

    def test_sftp_client_none_returns_false(
        self,
        uploader: SFTPUploader,
        mock_paramiko: MagicMock,
    ) -> None:
        """Test that sync returns False when SFTP client is None."""
        mock_paramiko.SFTPClient.from_transport.return_value = None

        with patch.dict(sys.modules, {"paramiko": mock_paramiko}):
            result = uploader.sync([("local.jpg", "remote.jpg")])

            assert result is False

    def test_cleanup_skips_invalid_date_files(
        self,
        uploader: SFTPUploader,
        mock_paramiko: MagicMock,
    ) -> None:
        """Test that cleanup skips files with invalid date format."""
        mock_sftp = mock_paramiko.SFTPClient.from_transport.return_value

        # Create file entries with invalid date formats
        invalid_entry = MagicMock()
        invalid_entry.filename = "webcam_invalid-date.avif"

        valid_entry = MagicMock()
        valid_entry.filename = "webcam_2020-01-01-00-00.avif"  # Old but valid

        mock_sftp.listdir_attr.return_value = [invalid_entry, valid_entry]
        mock_sftp.listdir.return_value = []

        with patch.dict(sys.modules, {"paramiko": mock_paramiko}):
            uploader.sync([])

            # Should only remove the valid old file, skip invalid
            assert mock_sftp.remove.call_count == 1
            remove_call_arg = mock_sftp.remove.call_args[0][0]
            assert "2020-01-01" in remove_call_arg

    def test_cleanup_skips_non_webcam_files(
        self,
        uploader: SFTPUploader,
        mock_paramiko: MagicMock,
    ) -> None:
        """Test that cleanup skips non-webcam files."""
        mock_sftp = mock_paramiko.SFTPClient.from_transport.return_value

        # Create file entries - some webcam, some not
        other_entry = MagicMock()
        other_entry.filename = "other_file.avif"

        txt_entry = MagicMock()
        txt_entry.filename = "webcam_2020-01-01-00-00.txt"  # Wrong extension

        mock_sftp.listdir_attr.return_value = [other_entry, txt_entry]
        mock_sftp.listdir.return_value = []

        with patch.dict(sys.modules, {"paramiko": mock_paramiko}):
            uploader.sync([])

            # Should not remove any files
            assert mock_sftp.remove.call_count == 0

    def test_upload_with_trailing_slash_in_path(
        self,
        mock_paramiko: MagicMock,
    ) -> None:
        """Test that trailing slash in path is handled correctly."""
        config = SFTPConfig(
            host="ftp.example.com",
            user="user",
            password="pass",
            path="/var/www/webcam/",  # Trailing slash
        )
        uploader = SFTPUploader(config)

        mock_sftp = mock_paramiko.SFTPClient.from_transport.return_value
        mock_sftp.listdir_attr.return_value = []
        mock_sftp.listdir.return_value = []

        with patch.dict(sys.modules, {"paramiko": mock_paramiko}):
            uploader.sync([("/tmp/test.jpg", "test.jpg")])

            # Path should not have double slashes
            put_call = mock_sftp.put.call_args[0][1]
            assert "//" not in put_call

    def test_default_retention_days(self) -> None:
        """Test that default retention_days is 7."""
        config = SFTPConfig()
        uploader = SFTPUploader(config)

        assert uploader.retention_days == 7

    def test_custom_retention_days(self) -> None:
        """Test that custom retention_days is applied."""
        config = SFTPConfig()
        uploader = SFTPUploader(config, retention_days=30)

        assert uploader.retention_days == 30
