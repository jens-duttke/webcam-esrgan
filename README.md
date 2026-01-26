# Webcam Interval Capture

Interval-based image capture and enhancement for IP cameras. Captures snapshots at configurable intervals and optionally enhances image quality by transferring high-frequency details from daytime reference images to nighttime captures.

## How It Works

Instead of using AI to generate synthetic details (like traditional super-resolution), this tool transfers **real details** from a daytime reference image to nighttime captures:

1. **Capture**: Image is captured in original camera resolution (e.g., 4K)
2. **Reference Selection**: Uses a configured daytime reference image for detail transfer
3. **Detail Transfer**: Uses Discrete Wavelet Transform (DWT) to transfer high-frequency details (textures, edges) from the reference to the current image
4. **Preserve Atmosphere**: Nighttime illumination, colors, and atmosphere are preserved - only details are enhanced
5. **Dual Output**: AVIF saved in original resolution (4K), JPEG resized for web display

### Why This Approach?

- **Real details**: Uses actual scene details instead of AI-hallucinated textures
- **Scene-aware**: Automatically adapts to scene brightness (more enhancement at night, less during day)
- **Fast**: No GPU required, pure CPU-based wavelet processing
- **Lightweight**: No large AI models to download or load

## Features

- Captures snapshots from IP cameras (tested with Reolink RLC-811WA)
- DWT-based detail transfer from daytime reference images
- Automatic brightness-adaptive enhancement strength
- Dual output format:
  - AVIF: Original resolution (4K) for archive/history
  - JPEG: Resized with timestamp for web display
- Optional SFTP upload for web publishing
- Live preview window with original/enhanced comparison
- Configurable capture intervals aligned to clock

## Requirements

- Python 3.10 or higher
- IP camera with HTTP snapshot support

## Installation

```bash
# Clone the repository
git clone https://github.com/jens-duttke/webcam-interval-capture.git
cd webcam-interval-capture

# Create virtual environment
python -m venv .

# Activate (Windows)
Scripts\activate

# Activate (Linux/macOS)
source bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

All settings are configured in `.env.local`:

```bash
cp .env.example .env.local
```

### Camera Settings (required)

| Variable | Example | Description |
|----------|---------|-------------|
| `CAMERA_IP` | `192.168.1.100` | IP address of your camera |
| `CAMERA_USER` | `admin` | Camera username |
| `CAMERA_PASSWORD` | `password` | Camera password |
| `CAMERA_CHANNEL` | `0` | Camera channel (usually 0) |

**Finding your camera channel:** To check the current channel assignment on your Reolink camera, open this URL in your browser:
```
https://<CAMERA_IP>/cgi-bin/api.cgi?cmd=GetChannelStatus&user=<CAMERA_USER>&password=<CAMERA_PASSWORD>
```

### Zoom/Focus Control (optional)

These settings ensure correct zoom and focus before each capture. Leave unset to skip verification.

**Note:** Zoom/Focus control requires a camera user with admin privileges.

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_ZOOM` | - | Expected zoom position (exact match required) |
| `CAMERA_FOCUS` | - | Expected focus position (with tolerance) |
| `CAMERA_FOCUS_TOLERANCE` | `5` | Acceptable deviation for focus position |

When configured, the script:
1. Queries current zoom/focus via Reolink API before each capture
2. If values don't match: sends adjustment commands and waits up to 30 seconds
3. If adjustment times out: skips the capture and retries at next interval

### Capture Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CAPTURE_INTERVAL` | `1` | Minutes between captures (clock-aligned, see below) |
| `TARGET_HEIGHT` | `1080` | Output height for JPEG in pixels (AVIF keeps original) |
| `SHOW_PREVIEW` | `true` | Show preview window (`true`/`false`) |
| `RETENTION_DAYS` | `7` | Days to keep history images on server |
| `OUTPUT_DIR` | `images` | Local directory for saved images |
| `TIMESTAMP_FORMAT` | `%Y-%m-%d %H:%M:%S` | Format for timestamp overlay (strftime) |

#### Clock-Aligned Capture Intervals

The `CAPTURE_INTERVAL` setting uses **clock-aligned timing** based on midnight (00:00). This means captures always occur at "round" times, regardless of when the script was started.

**How it works:**
- Intervals are calculated from 00:00 (midnight), not from script start time
- The script waits until the next aligned time before the first capture
- This ensures consistent, predictable capture times

**Examples:**

| Interval | Script Start | First Capture | Subsequent Captures |
|----------|--------------|---------------|---------------------|
| `1` | 16:32:45 | 16:33:00 | 16:34, 16:35, 16:36... |
| `5` | 16:32:45 | 16:35:00 | 16:40, 16:45, 16:50... |
| `10` | 16:32:45 | 16:40:00 | 16:50, 17:00, 17:10... |
| `15` | 16:32:45 | 16:45:00 | 17:00, 17:15, 17:30... |
| `30` | 16:32:45 | 17:00:00 | 17:30, 18:00, 18:30... |

This clock-alignment makes it easy to predict when captures will occur and ensures that images from different days are taken at exactly the same times (e.g., always at :00, :10, :20, :30, :40, :50 for a 10-minute interval).

### Detail Transfer Enhancement Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENHANCE_MAX_STRENGTH` | `0.15` | Maximum strength for dark images (0-1) |
| `ENHANCE_BRIGHTNESS_THRESHOLD` | `0.3` | Only enhance below this brightness (0-1) |
| `ENHANCE_WAVELET` | `db4` | Wavelet type: `db4`, `haar`, `sym4`, `bior1.3` |
| `ENHANCE_LEVELS` | `3` | DWT decomposition levels (2-4 recommended) |
| `ENHANCE_FUSION_MODE` | `weighted` | Fusion mode: `weighted` (brightness-based) or `max_energy` (edge-based) |

### Reference Image Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `REFERENCE_PATH` | - | Fixed reference image path (leave empty for auto-select) |
| `REFERENCE_DIR` | `archive` | Directory for reference images (auto-selection) |
| `REFERENCE_HOUR` | `12` | Hour to select reference from (0-23) |

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
| `webcam_current.jpg` | JPEG | Latest image, resized to `TARGET_HEIGHT` with timestamp |
| `webcam_current.avif` | AVIF | Latest image in original resolution (no timestamp) |
| `webcam_YYYY-MM-DD-HH-MM.avif` | AVIF | Timestamped history images in original resolution |
| `webcam_log.json` | JSON | List of all history images (for web integration) |

## Usage

```bash
# Run with virtual environment
Scripts/python main.py

# Or if installed as package
webcam-interval-capture
```

### Preview Controls

- **Click and hold**: Show original image
- **Spacebar**: Toggle between original/enhanced
- **Q or ESC**: Exit program
- **X button**: Close preview (program continues)
- **W**: Reopen preview window

## Workflow

1. Waits for the next clock-aligned interval (e.g., :00, :10, :20... for 10-minute interval)
2. Verifies zoom/focus settings (if configured), adjusts if necessary
3. Fetches snapshot from camera via HTTP
4. Computes auto-strength based on image brightness (dark = more enhancement)
5. Applies DWT-based detail transfer from reference image (if configured)
6. Saves JPEG (resized to `TARGET_HEIGHT` with timestamp)
7. Saves AVIF (original resolution, no timestamp) for history
8. Uploads to SFTP server (if configured)
9. Cleans up images older than `RETENTION_DAYS`
10. Updates `webcam_log.json` with file list

## Algorithm

The detail transfer uses a multi-scale Discrete Wavelet Transform (DWT) approach:

1. **Decomposition**: Both images are decomposed into approximation (low-frequency) and detail (high-frequency) coefficients using the specified wavelet
2. **Approximation Preservation**: The approximation coefficients always come from the nighttime image, preserving overall illumination
3. **Detail Fusion**: Detail coefficients are selectively combined based on:
   - **Weighted mode**: Blends details based on local brightness (dark areas get more reference details)
   - **Max-energy mode**: Selects whichever image has stronger local edges
4. **Reconstruction**: The fused coefficients are reconstructed into the final image

### Auto-Strength

The enhancement strength is automatically computed based on image brightness:
- Dark images (night): Up to `ENHANCE_MAX_STRENGTH`
- Bright images (day): Approaches 0 (minimal enhancement)

This ensures daytime images remain largely unchanged while nighttime images receive appropriate detail enhancement.

## File Structure

```
webcam-interval-capture/
├── main.py                     # Main application
├── webcam_interval_capture/    # Package
│   ├── archive.py              # Archive and reference management
│   ├── camera.py               # Camera snapshot handling
│   ├── config.py               # Configuration management
│   ├── enhance.py              # DWT detail transfer
│   ├── image.py                # Image processing/saving
│   └── sftp.py                 # SFTP upload
├── tests/                      # Test suite
├── images/                     # Output images (created automatically)
├── .env.example                # Configuration template
└── requirements.txt            # Dependencies
```

## Development

```bash
# Install dev dependencies
pip install pytest pytest-cov ruff mypy

# Run tests
Scripts/python -m pytest -v

# Lint
Scripts/python -m ruff check .

# Format
Scripts/python -m ruff format .

# Type check
Scripts/python -m mypy webcam_interval_capture/
```

## License

MIT License
