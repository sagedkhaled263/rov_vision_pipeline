# Triton ROV — Vision System

Computer vision software for the **Triton (Shadowing) ROV** — MATE ROV competition, AAST-MT.  
Covers three tasks: **underwater length measurement**, **invasive crab detection**, and **3D model video capture**.

---

## Repository Structure

```
triton-rov-vision/
├── crab_detection/
│   ├── receiver_crab_detection.py  # Laptop-side: receives frames, runs YOLOv11 detection
│   └── crab_detection.pt             # YOLOv11 model weights — add manually after cloning
│
├── length_measurement/
│   └── measuring_length.py         # ROV-side: OAK-D stereo depth + web HUD (runs headless)
│
└── 3d_recording/
    ├── oak_d_sender_pi.py          # Pi-side: streams 1080p RGB from OAK-D over TCP
    ├── oak_d_receiver_laptop.py    # Laptop-side: receives stream, records MP4 for 3D reconstruction
    └── README_Video_to_3D.pdf      # Full reconstruction pipeline (COLMAP + Instant-NGP)
```

---

## Tasks

### 1. Length Measurement (`length_measurement/`)

**Runs on the ROV** (OAK-D connected via USB).  
Uses stereo depth + pinhole back-projection to measure the real-world distance between two user-clicked points. Serves a browser-based HUD accessible from any laptop on the same network.

**How it works:**
- OAK-D stereo pipeline produces aligned RGB + depth at 640×360
- User freezes the feed (Spacebar), clicks two points on the browser HUD
- Server back-projects pixel coordinates using camera intrinsics and depth ROI
- Applies water-depth correction (refractive index offset, calibrated by depth)
- Applies distance-aware edge correction (alpha) to compensate for wide-angle barrel distortion at frame edges
- Returns `distance ± uncertainty` in cm

**Modes:**
| Mode | Description |
|------|-------------|
| `DIRECT 3D` | Click two points → immediate 3D measurement |
| `SET REFERENCE` | Click a known-length object first to calibrate a scale factor, then measure an unknown |

**Configuration (top of `measuring_length.py`):**

```python
UNDERWATER_MODE = True   # True = flashed EEPROM calibration, False = factory
PORT = 1211              # Browser connects to http://<ROV_IP>:1211
JPEG_QUALITY = 70
```

**Water depth correction table** — editable live from the HUD or hard-coded in `WATER_DEPTH_CORRECTION`. Format: `(depth_m, correction_factor)` where `correction_factor = true_length / measured_length`.

**Run:**
```bash
# On the ROV (OAK-D plugged in)
python3 length_measurement/measuring_length.py

# On any laptop — open browser
http://<ROV_IP>:1211
```

**Dependencies:**
```
depthai
opencv-python
numpy
```

---

### 2. Crab Detection (`crab_detection/`)

Detects **European green crab** (*Carcinus maenas*) — an invasive species — in real-time using YOLOv8.

**Architecture:**  
- Frame sender runs on the ROV (separate sender script, not in this repo — frames sent over TCP port 9002)
- `receiver.py` runs on the **laptop**, connects to the ROV, pulls frames, runs inference locally, and displays results in a window

**Configuration (top of `receiver_crab_detection.py`):**

```python
MODEL_PATH      = "Latest_Model.pt"   # Place model in crab_detection/
SENDER_HOST     = "192.168.1.100"     # ROV IP
PORT            = 9002
CONF_THRESHOLD  = 0.8
```

**Run:**
```bash
# Drop Latest_Model.pt into crab_detection/, then:
cd crab_detection
python3 receiver_crab_detection.py

# Press Q to quit
```

**Dependencies:**
```
ultralytics
opencv-python
numpy
```

> **Note:** `Latest_Model.pt` must be present in `crab_detection/` before running.

---

### 3. 3D Model Video Capture (`3d_recording/`)

Records high-quality 1080p video from the OAK-D RGB camera for offline 3D reconstruction (photogrammetry / NeRF).

**Two-part pipeline:**

| Script | Runs on | Role |
|--------|---------|------|
| `oak_d_sender_pi.py` | Raspberry Pi (inside ROV) | Streams 1080p JPEG over TCP port 5001 |
| `oak_d_receiver_laptop.py` | Laptop | Receives stream, records MP4 on demand |

**Run:**
```bash
# On the Pi
python3 3d_recording/oak_d_sender_pi.py

# On the laptop
python3 3d_recording/oak_d_receiver_laptop.py
# SPACE = start/stop recording | Q = quit
# Videos saved to ~/3d_video/
```

**Full reconstruction pipeline** (COLMAP + Instant-NGP → mesh export): see **[`3d_recording/README_Video_to_3D.pdf`](3d_recording/README Video to 3D.pdf)**

---

**Dependencies (Pi):**
```
depthai
opencv-python
```

**Dependencies (Laptop):**
```
opencv-python
numpy
```

---

## Network Setup

All scripts assume the ROV/Pi is at `192.168.1.100`. Adjust the `SENDER_HOST` / `PI_IP` constants if your subnet differs.

| Service | Host | Port |
|---------|------|------|
| Length measurement HUD | ROV | 1211 |
| Crab detection stream | ROV | 9002 |
| 3D recording stream | Pi | 5001 |

---



## Hardware

- **OAK-D** (DepthAI) — stereo depth + RGB camera
- **Raspberry Pi** — onboard compute for video streaming
- **Laptop** — runs detection inference, browser HUD, and video recording

---

*MATE ROV Competition — Triton / Shadowing ROV Team, AAST-MT*
