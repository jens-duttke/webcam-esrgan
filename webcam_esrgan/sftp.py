"""SFTP upload functionality."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramiko import SFTPClient

from webcam_esrgan.config import SFTPConfig


class SFTPUploader:
    """Handles SFTP upload, cleanup, and log management."""

    def __init__(
        self,
        config: SFTPConfig,
        retention_days: int = 7,
        capture_interval: int = 1,
    ) -> None:
        """
        Initializes the SFTP uploader.

        Args:
            config: SFTP connection settings.
            retention_days: Days to keep old images before cleanup.
            capture_interval: Interval in minutes between captures.
        """
        self.config = config
        self.retention_days = retention_days
        self.capture_interval = capture_interval

    def sync(self, files_to_upload: list[tuple[str, str]]) -> bool:
        """
        Performs all SFTP operations in a single connection.

        Uploads new files, cleans up old files, and updates the JSON log.

        Args:
            files_to_upload: List of (local_path, remote_filename) tuples.

        Returns:
            True if sync succeeded, False otherwise.
        """
        if not self.config.enabled:
            return False

        try:
            import paramiko

            transport = paramiko.Transport((self.config.host, self.config.port))
            transport.connect(
                username=self.config.user,
                password=self.config.password,
            )
            sftp = paramiko.SFTPClient.from_transport(transport)

            if sftp is None:
                print("  SFTP Error: Could not create SFTP client")
                return False

            remote_dir = self.config.path.rstrip("/")  # type: ignore[union-attr]

            try:
                self._upload_files(sftp, files_to_upload, remote_dir)
                self._cleanup_old_files(sftp, remote_dir)
                self._update_log(sftp, remote_dir)
                return True
            finally:
                sftp.close()
                transport.close()

        except ImportError:
            print("  SFTP Error: paramiko not installed (pip install paramiko)")
            return False
        except Exception as e:
            print(f"  SFTP Error: {e}")
            return False

    def _upload_files(
        self,
        sftp: SFTPClient,
        files: list[tuple[str, str]],
        remote_dir: str,
    ) -> None:
        """Uploads files to the SFTP server."""
        for local_path, remote_filename in files:
            remote_path = f"{remote_dir}/{remote_filename}"
            sftp.put(local_path, remote_path)
            print(f"Uploaded: {remote_filename}")

    def _cleanup_old_files(self, sftp: SFTPClient, remote_dir: str) -> None:
        """Deletes files older than retention_days."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0

        for entry in sftp.listdir_attr(remote_dir):
            filename = entry.filename

            # Only process timestamped webcam files
            if not filename.startswith("webcam_") or filename.startswith(
                "webcam_current."
            ):
                continue
            if not (filename.endswith(".avif") or filename.endswith(".jpg")):
                continue

            # Extract date from filename: webcam_YYYY-MM-DD-HH-MM.avif/.jpg
            try:
                base = filename[7:]  # Remove 'webcam_'
                date_str = base.rsplit(".", 1)[0]  # Remove extension
                file_date = datetime.strptime(date_str, "%Y-%m-%d-%H-%M")
                if file_date < cutoff_date:
                    sftp.remove(f"{remote_dir}/{filename}")
                    deleted_count += 1
            except ValueError:
                pass  # Skip files with invalid date format

        if deleted_count > 0:
            print(f"Cleaned up: {deleted_count} old files removed")

    def _update_log(self, sftp: SFTPClient, remote_dir: str) -> None:
        """Updates webcam_log.json with list of all history images."""
        image_files: list[str] = []

        for entry in sftp.listdir(remote_dir):
            is_history_file = entry.startswith("webcam_") and not entry.startswith(
                "webcam_current."
            )
            is_image = entry.endswith(".avif") or entry.endswith(".jpg")
            if is_history_file and is_image:
                image_files.append(entry)

        image_files.sort()
        log_data = {
            "captureInterval": self.capture_interval,
            "images": image_files,
        }
        json_content = json.dumps(log_data, indent=2)

        with sftp.file(f"{remote_dir}/webcam_log.json", "w") as f:
            f.write(json_content)

        print(f"Log updated: {len(image_files)} images")
