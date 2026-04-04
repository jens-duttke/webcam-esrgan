"""Rsync over SSH upload functionality.

Compatible with rrsync (restricted rsync) setups — all operations
(upload, listing, cleanup) use the rsync protocol, not SSH commands.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from webcam_interval_capture.config import RsyncConfig


class RsyncUploader:
    """Handles file upload, cleanup, and log management via rsync over SSH."""

    def __init__(
        self,
        config: RsyncConfig,
        retention_days: int = 7,
        capture_interval: int = 1,
    ) -> None:
        """
        Initializes the rsync uploader.

        Args:
            config: Rsync/SSH connection settings.
            retention_days: Days to keep old images before cleanup.
            capture_interval: Interval in minutes between captures.
        """
        self.config = config
        self.retention_days = retention_days
        self.capture_interval = capture_interval

    def sync(self, files_to_upload: list[tuple[str, str]]) -> bool:
        """
        Uploads files via rsync, cleans up old files, and updates the log.

        Args:
            files_to_upload: List of (local_path, remote_filename) tuples.

        Returns:
            True if sync succeeded, False otherwise.
        """
        if not self.config.enabled:
            return False

        try:
            assert self.config.host is not None
            assert self.config.user is not None
            assert self.config.ssh_key is not None
            assert self.config.path is not None

            remote_dir = self.config.path.rstrip("/")

            self._upload_files(files_to_upload, remote_dir)

            remote_files = self._list_remote_files(remote_dir)
            deleted = self._cleanup_old_files(remote_dir, remote_files)
            remaining = [f for f in remote_files if f not in deleted]

            self._update_log(remote_dir, remaining)
            return True

        except Exception as e:
            print(f"  Rsync Error: {e}")
            return False

    def _resolve_bin(self, name: str) -> str:
        """Returns full path to a binary, using bin_path if configured."""
        if self.config.bin_path:
            return str(Path(self.config.bin_path) / name)
        return name

    def _ssh_remote(self) -> str:
        """Returns user@host connection string."""
        return f"{self.config.user}@{self.config.host}"

    def _rsync_base_cmd(self) -> list[str]:
        """Returns the common rsync command prefix with SSH transport."""
        ssh_key = self.config.ssh_key
        port = self.config.port
        ssh_bin = self._resolve_bin("ssh").replace("\\", "/")
        ssh_key_posix = ssh_key.replace("\\", "/") if ssh_key else ""
        ssh_cmd = (
            f'"{ssh_bin}" -i "{ssh_key_posix}" -p {port}'
            " -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
        )
        return [
            self._resolve_bin("rsync"),
            "-e",
            ssh_cmd,
        ]

    def _upload_files(
        self,
        files: list[tuple[str, str]],
        remote_dir: str,
    ) -> None:
        """Uploads files to the remote server via rsync in a single call."""
        if not files:
            return

        local_paths: list[str] = []
        for local_path, _remote_filename in files:
            local_paths.append(local_path.replace("\\", "/"))

        cmd = self._rsync_base_cmd() + [
            "-az",
            "--chmod=F644",
            *local_paths,
            f"{self._ssh_remote()}:{remote_dir}/",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  Rsync upload failed: {result.stderr.strip()}")
        else:
            for _local_path, remote_filename in files:
                print(f"Uploaded (rsync): {remote_filename}")

    def _list_remote_files(self, remote_dir: str) -> list[str]:
        """
        Lists files in the remote directory via rsync --list-only.

        Args:
            remote_dir: Remote directory path.

        Returns:
            List of filenames in the remote directory.
        """
        cmd = self._rsync_base_cmd() + [
            "--list-only",
            f"{self._ssh_remote()}:{remote_dir}/",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return []

        # rsync --list-only output format: "drwxr-xr-x  123 2026/01/20 10:00:00 name"
        # The filename is the last field after the date/time columns
        files: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("d"):
                continue  # Skip directories
            parts = line.split(None, 4)
            if len(parts) >= 5:
                files.append(parts[4])
        return files

    def _cleanup_old_files(
        self,
        remote_dir: str,
        remote_files: list[str],
    ) -> set[str]:
        """
        Deletes files older than retention_days via rsync --delete.

        Uses rsync with --include/--exclude filters to selectively
        remove old files from the remote directory.

        Args:
            remote_dir: Remote directory path.
            remote_files: List of filenames currently on the server.

        Returns:
            Set of filenames that were deleted.
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        to_delete: list[str] = []

        for filename in remote_files:
            if not filename.startswith("webcam_") or filename.startswith(
                "webcam_current."
            ):
                continue
            if not (filename.endswith(".avif") or filename.endswith(".jpg")):
                continue

            try:
                base = filename[7:]  # Remove 'webcam_'
                date_str = base.rsplit(".", 1)[0]  # Remove extension
                file_date = datetime.strptime(date_str, "%Y-%m-%d-%H-%M")
                if file_date < cutoff_date:
                    to_delete.append(filename)
            except ValueError:
                pass  # Skip files with invalid date format

        if to_delete:
            # Use rsync --delete with --include/--exclude to remove specific files.
            # Strategy: sync an empty dir, include only the files to delete,
            # exclude everything else, with --delete to remove matched files.
            empty_dir = tempfile.mkdtemp(dir=".")
            # Convert to relative path — absolute Windows paths (e.g. D:/...)
            # make rsync interpret the drive letter as a remote host.
            empty_dir_rel = os.path.relpath(empty_dir).replace("\\", "/")
            try:
                # Use = syntax to prevent MSYS2/Windows glob expansion
                # of the * wildcard in --exclude
                cmd = self._rsync_base_cmd() + ["-r", "--delete"]
                for f in to_delete:
                    cmd.append(f"--include={f}")
                cmd += [
                    "--exclude=*",
                    empty_dir_rel + "/",
                    f"{self._ssh_remote()}:{remote_dir}/",
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    print(f"Cleaned up (rsync): {len(to_delete)} old files removed")
                else:
                    print(f"  Rsync cleanup failed: {result.stderr.strip()}")
                    to_delete = []  # Nothing was deleted
            except subprocess.TimeoutExpired:
                print("  Rsync cleanup timed out, skipping")
                to_delete = []
            finally:
                Path(empty_dir).rmdir()

        return set(to_delete)

    def _update_log(
        self,
        remote_dir: str,
        remote_files: list[str],
    ) -> None:
        """
        Updates webcam_log.json on the remote server.

        Args:
            remote_dir: Remote directory path.
            remote_files: List of filenames currently on the server.
        """
        image_files: list[str] = []

        for entry in remote_files:
            is_history_file = entry.startswith("webcam_") and not entry.startswith(
                "webcam_current."
            )
            is_image = entry.endswith(".avif") or entry.endswith(".jpg")
            if is_history_file and is_image:
                image_files.append(entry)

        image_files.sort()
        log_data: dict[str, int | list[str]] = {
            "captureInterval": self.capture_interval,
            "images": image_files,
        }

        tmp_path = Path("webcam_log.tmp.json")
        try:
            tmp_path.write_text(json.dumps(log_data, indent=2))

            remote_path = f"{remote_dir}/webcam_log.json"
            cmd = self._rsync_base_cmd() + [
                "-az",
                "--chmod=F644",
                str(tmp_path),
                f"{self._ssh_remote()}:{remote_path}",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                print(f"Log updated (rsync): {len(image_files)} images")
            else:
                print(f"  Rsync log upload failed: {result.stderr.strip()}")
        finally:
            tmp_path.unlink(missing_ok=True)
