# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
