# Webcam Enhancement with Real-ESRGAN

Captures snapshots from IP cameras and enhances image quality using Real-ESRGAN AI upscaling. Supports automatic SFTP upload for web integration.

## Features

- HTTP snapshot capture from IP cameras (Reolink API)
- AI-based image enhancement using Real-ESRGAN
- Dual output: JPEG for current image, AVIF for history (smaller file size)
- Automatic SFTP upload to web server
- Configurable image retention with automatic cleanup
- JSON log file for web integration
- Clock-synchronized captures (aligned to minute intervals)
- All settings configurable via environment variables

## Requirements

- Python 3.10-3.12 (3.13+ has compatibility issues with basicsr)
- IP camera with HTTP snapshot API (tested with Reolink)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jens-duttke/webcam-esrgan.git
   cd webcam-esrgan
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

3. Install PyTorch (choose one):
   ```bash
   # CPU only
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # CUDA 12.1 (for GPU acceleration)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create configuration file:
   ```bash
   cp .env.example .env.local
   ```

6. Edit `.env.local` with your settings (see Configuration below).

## Usage

```bash
python main.py
```

To stop: Close the preview window, press `q`, `ESC`, or `Ctrl+C`.

### Preview Window Controls

When `SHOW_PREVIEW=true` (default), a preview window displays the enhanced image. You can compare it with the original unprocessed image using these controls:

| Control | Action |
|---------|--------|
| **Mouse click (hold)** | Shows original image while held |
| **Spacebar** | Toggles between original and enhanced image |
| **q** or **ESC** | Closes the application |

The preview window:
- Opens at 600 pixels height with correct aspect ratio
- Can be freely resized; image maintains aspect ratio with letterboxing (black bars)
- Both original and enhanced images are scaled to fit the window

## Configuration

All settings are configured in `.env.local`:

### Camera Settings (required)

| Variable | Example | Description |
|----------|---------|-------------|
| `CAMERA_IP` | `192.168.1.100` | IP address of your camera |
| `CAMERA_USER` | `admin` | Camera username |
| `CAMERA_PASSWORD` | `password` | Camera password |
| `CAMERA_CHANNEL` | `0` | Camera channel (usually 0) |

### Capture Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CAPTURE_INTERVAL` | `1` | Minutes between captures (aligned to clock) |
| `TARGET_HEIGHT` | `1080` | Output resolution height in pixels |
| `SHOW_PREVIEW` | `true` | Show preview window (`true`/`false`) |
| `RETENTION_DAYS` | `7` | Days to keep history images on server |
| `OUTPUT_DIR` | `images` | Local directory for saved images |
| `TIMESTAMP_FORMAT` | `%Y-%m-%d %H:%M:%S` | Format for timestamp overlay (strftime) |

### AI Enhancement Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSCALE_FACTOR` | `3` | AI upscale factor (2 = more original detail, 4 = more AI detail) |
| `ENHANCEMENT_BLEND` | `0.8` | Blend ratio: 0.0 = original only, 1.0 = AI only |

### Image Quality Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `JPEG_QUALITY` | `80` | JPEG quality for current image (1-100) |
| `AVIF_QUALITY` | `65` | AVIF quality for history images (1-100) |
| `AVIF_SPEED` | `4` | AVIF encoding speed (0 = slow/best, 10 = fast) |
| `AVIF_SUBSAMPLING` | `4:2:0` | Chroma subsampling (`4:4:4`, `4:2:2`, `4:2:0`) |

### SFTP Upload (optional)

Leave `SFTP_HOST` empty to disable upload.

| Variable | Example | Description |
|----------|---------|-------------|
| `SFTP_HOST` | `ftp.example.com` | SFTP server hostname |
| `SFTP_PORT` | `22` | SFTP port |
| `SFTP_USER` | `username` | SFTP username |
| `SFTP_PASSWORD` | `password` | SFTP password |
| `SFTP_PATH` | `/httpdocs/webcam` | Remote directory path |

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `webcam_current.jpg` | JPEG | Always the latest image (for webcam services) |
| `webcam_YYYY-MM-DD-HH-MM.avif` | AVIF | Timestamped history images |
| `webcam_log.json` | JSON | List of all history images (for web integration) |

## How It Works

1. Waits for the next full minute (clock-synchronized)
2. Fetches snapshot from camera via HTTP
3. Shrinks image to `TARGET_HEIGHT / UPSCALE_FACTOR`
4. Upscales using Real-ESRGAN AI
5. Blends AI result with original (configurable ratio)
6. Adds timestamp overlay
7. Saves as JPEG (current) and AVIF (history)
8. Uploads to SFTP server (if configured)
9. Cleans up images older than `RETENTION_DAYS`
10. Updates `webcam_log.json` with file list

## AI Model

Uses `realesr-general-x4v3.pth` (~5MB), automatically downloaded on first run.

The model uses SRVGGNetCompact architecture, optimized for general image enhancement.

## Project Structure

```
webcam-esrgan/
├── main.py                 # Application entry point
├── webcam_esrgan/          # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── camera.py           # Camera snapshot capture
│   ├── enhance.py          # Real-ESRGAN enhancement
│   ├── sftp.py             # SFTP upload functionality
│   └── image.py            # Image processing utilities
├── tests/                  # Unit tests
│   ├── test_config.py
│   ├── test_camera.py
│   ├── test_sftp.py
│   └── test_image.py
├── .github/workflows/      # CI/CD
│   └── tests.yml
├── .env.example            # Configuration template
├── requirements.txt        # Dependencies
└── pyproject.toml          # Project metadata
```

## Development

### Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=webcam_esrgan
```

### Linting

```bash
pip install ruff
ruff check .
ruff format .
```

### Type Checking

```bash
pip install mypy
mypy webcam_esrgan/
```

## Example .env.local

```env
# Camera
CAMERA_IP=192.168.178.71
CAMERA_USER=webcam
CAMERA_PASSWORD=secret123
CAMERA_CHANNEL=0

# Capture
CAPTURE_INTERVAL=1
TARGET_HEIGHT=1080
SHOW_PREVIEW=false
RETENTION_DAYS=7
TIMESTAMP_FORMAT=%d.%m.%Y %H:%M

# AI
UPSCALE_FACTOR=3
ENHANCEMENT_BLEND=0.8

# Quality
JPEG_QUALITY=80
AVIF_QUALITY=65
AVIF_SPEED=4

# Output
OUTPUT_DIR=images

# SFTP
SFTP_HOST=ftp.mywebsite.com
SFTP_PORT=22
SFTP_USER=webcam_user
SFTP_PASSWORD=secret456
SFTP_PATH=/httpdocs/webcam
```

## License

MIT
