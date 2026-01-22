# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Interactive preview window with original/enhanced image comparison
  - Click and hold mouse button to show original image
  - Press spacebar to toggle between original and enhanced view
  - Press `q` or `ESC` to close the application
- Preview window now opens at 600px height with correct aspect ratio (based on image)
- Preview window uses letterboxing to maintain aspect ratio when resized

### Changed

- Timestamp overlay now only on JPEG files; AVIF files are saved without overlay
- Python version requirement restricted to 3.10-3.12 (basicsr incompatible with 3.13+)

### Fixed

- Excluded `webcam_current.avif` from `webcam_log.json` (only history files should be listed)
- SSL certificate verification error on Windows when downloading Real-ESRGAN model (now uses certifi)

## [1.0.0] - 2026-01-21

### Added

- Initial release
- HTTP snapshot capture from IP cameras (Reolink API)
- AI-based image enhancement using Real-ESRGAN (SRVGGNetCompact architecture)
- Automatic GPU detection (CUDA if available, CPU fallback)
- Dual output format: JPEG for current image, AVIF for history
- Automatic SFTP upload to web server with single-connection optimization
- Configurable image retention with automatic cleanup
- JSON log file generation for web integration
- Clock-synchronized captures aligned to minute intervals
- Timestamp overlay on enhanced images
- All settings configurable via `.env.local` file
- Modular package structure (`webcam_esrgan/`)
- Comprehensive test suite with 100% code coverage
- CI/CD with GitHub Actions (Python 3.10, 3.11, 3.12)
- Type hints throughout with mypy validation
- Code linting with ruff
