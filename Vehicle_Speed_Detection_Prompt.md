# Multi-Vehicle Speed Detection System — Detailed Build Prompt

> Copy the entire prompt below and paste it into Claude, ChatGPT, or any AI coding assistant.
> It will generate the complete working project for you.
>
> **Reference project:** https://github.com/ISHANT-AG/BengaluruTrafficAI
> This prompt takes the speed estimation concept from that project and builds it as a standalone, simplified, exhibition-ready system.

---

## THE PROMPT — START COPYING FROM HERE

---

Build me a complete **Multi-Vehicle Speed Detection System** in Python that can detect and estimate the speed of multiple vehicles simultaneously from a traffic video clip. This is for a college AI/ML project exhibition, so keep the code beginner-friendly with clear comments on every section. I'll describe every single detail below — follow it exactly.

**IMPORTANT REFERENCE:** I previously built a larger traffic AI system at https://github.com/ISHANT-AG/BengaluruTrafficAI which included speed estimation as one feature. That project used YOLOv8n for detection, a pure-Python IoU multi-object tracker with persistent vehicle IDs, and a pixel-to-meter velocity calculator. I want you to take ONLY the speed detection concept from it and build it as a clean standalone project. Do NOT copy that repo's code. Build fresh, simpler, and focused only on multi-vehicle speed detection.

---

### PROJECT OVERVIEW

The system takes a traffic video as input (recorded from a fixed CCTV-style camera angle — either a real traffic video or a dashcam clip). It detects every vehicle in each frame using YOLOv8, tracks them across frames with unique IDs, and calculates their speed in km/h based on how far they travel between two virtual reference lines drawn on the video. The output is the same video with bounding boxes, vehicle IDs, and speed labels drawn on every tracked vehicle in real-time.

---

### HOW SPEED ESTIMATION WORKS (The core logic — implement this exactly)

The approach uses two horizontal virtual lines drawn across the road in the video frame. The real-world distance between where these two lines map on the actual road is known (or estimated). When a vehicle crosses Line 1, we record the timestamp. When the same vehicle (same tracking ID) crosses Line 2, we record that timestamp too. Speed = known real-world distance / time difference.

```
VIDEO FRAME:
┌─────────────────────────────────────────────────┐
│                                                 │
│     🚗 car_id=3  (not yet crossed any line)     │
│                                                 │
│ ═══════════ LINE 1 (Entry) ═══════════════════  │  ← vehicle crosses here → record time_in
│                                                 │
│     🚙 car_id=1  speed: calculating...          │
│                                                 │
│     🚕 car_id=2  speed: 47 km/h                 │
│                                                 │
│ ═══════════ LINE 2 (Exit) ════════════════════  │  ← vehicle crosses here → record time_out
│                                                 │
│     🚗 car_id=0  speed: 52 km/h  ✓ done        │
│                                                 │
└─────────────────────────────────────────────────┘

Speed of vehicle = KNOWN_DISTANCE_METERS / (time_out - time_in) × 3.6
                   (× 3.6 converts m/s to km/h)
```

This two-line method is much more accurate than per-frame pixel displacement because it anchors to a known real-world measurement and is not affected by camera perspective distortion as heavily.

---

### TECH STACK (Use only these — nothing else)

- `ultralytics` — for YOLOv8n vehicle detection (pre-trained, no custom training needed)
- `opencv-python (cv2)` — for video reading, frame processing, drawing overlays, saving output video
- `numpy` — for array operations and distance calculations
- `collections` — for defaultdict and deque to track vehicle histories
- `time` — for timestamps
- `argparse` — for command-line arguments
- `math` — for distance calculations
- `csv` — for saving speed logs

At the very top of the main file, include this comment block:

```python
# ============================================================
# Multi-Vehicle Speed Detection System
# ============================================================
# Team: [Team Name]
# Project Exhibition 2026
#
# Installation:
#   pip install ultralytics opencv-python numpy
#
# Usage:
#   python speed_detector.py --video traffic.mp4
#   python speed_detector.py --video traffic.mp4 --output result.mp4
#
# How it works:
#   1. YOLOv8n detects vehicles in every frame
#   2. A simple centroid tracker assigns persistent IDs
#   3. Two virtual lines on the road measure crossing times
#   4. Speed = known distance / crossing time
# ============================================================
```

---

### FOLDER STRUCTURE

```
vehicle-speed-detector/
├── speed_detector.py          ← main program (single file, entire project)
├── tracker.py                 ← simple centroid tracker class (separate file)
├── videos/                    ← put sample traffic videos here
│   └── sample_traffic.mp4
├── output/                    ← auto-created, stores output videos
│   └── result_sample_traffic.mp4
├── logs/                      ← auto-created, stores speed CSV logs
│   └── speed_log_2026-07-27.csv
├── requirements.txt
└── README.md
```

Use `os.makedirs(path, exist_ok=True)` for each folder.

---

### FILE 1: `tracker.py` — Simple Centroid Tracker (Detailed)

Build a simple multi-object tracker from scratch. Do NOT use any external tracking library (no SORT, no DeepSORT, no ByteTrack). The point of the exhibition is to show we understand tracking, not that we can import a library.

**How the centroid tracker works (explain with comments in code):**

1. For each detected vehicle, calculate the centroid (center point) of its bounding box: `cx = (x1 + x2) / 2, cy = (y1 + y2) / 2`

2. If no objects are currently being tracked (first frame), register all detected centroids as new objects with unique IDs starting from 0.

3. For subsequent frames:
   - Calculate the Euclidean distance between every existing tracked centroid and every new detected centroid
   - Use a greedy matching approach: assign each new detection to the closest existing track, BUT only if the distance is below a threshold (default: 80 pixels). This prevents matching two vehicles on opposite sides of the road.
   - Any new detection that doesn't match an existing track → register as a new vehicle with a new ID
   - Any existing track that doesn't get a match for N consecutive frames (default: 30 frames) → deregister it (the vehicle probably left the frame)

4. Return a dictionary mapping `{vehicle_id: (cx, cy)}` for the current frame

**The class should look like this:**

```python
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        """
        max_disappeared: how many frames a vehicle can be missing before we stop tracking it
        max_distance: maximum pixel distance to consider two detections as the same vehicle
        """

    def register(self, centroid):
        """Register a new vehicle with the next available ID."""

    def deregister(self, object_id):
        """Remove a vehicle that has disappeared."""

    def update(self, detections):
        """
        Main method — called every frame.
        detections: list of bounding boxes [(x1, y1, x2, y2), ...]
        Returns: dict {vehicle_id: (cx, cy, x1, y1, x2, y2)}
        """
```

**Important:** Also return the bounding box alongside the centroid for each tracked vehicle, because we need it for drawing rectangles on screen. Store it inside the tracker as `self.bboxes[object_id] = (x1, y1, x2, y2)`.

This tracker is simpler than the IoU tracker in BengaluruTrafficAI. That's fine — for a fixed camera angle with vehicles moving in one direction, centroid distance matching works well enough.

---

### FILE 2: `speed_detector.py` — Main Program (Very Detailed)

This is the main file. It should be well-organized with clear sections.

#### SECTION 1: CONFIGURATION

Define all tunable values at the top as constants so they're easy to change:

```python
# ===== CONFIGURATION =====
# Adjust these based on your video

# YOLO settings
MODEL_NAME = "yolov8n.pt"            # YOLOv8 nano — fast and lightweight
CONFIDENCE_THRESHOLD = 0.4            # minimum detection confidence
VEHICLE_CLASSES = [2, 3, 5, 7]        # COCO class IDs: car=2, motorcycle=3, bus=5, truck=7

# Virtual line positions (as percentage of frame height, 0.0 = top, 1.0 = bottom)
# Adjust these based on your video — lines should be placed where vehicles
# travel roughly straight (no curves) and perpendicular to camera view
LINE_1_POSITION = 0.35               # entry line at 35% from top
LINE_2_POSITION = 0.65               # exit line at 65% from top

# Real-world distance between the two lines IN METERS
# This is the actual road distance between where Line 1 and Line 2 map in reality
# For a sample video, estimate this from lane markings (standard lane marking gap = 3m in India)
# If you have 4 marking gaps visible between the lines, KNOWN_DISTANCE = 12.0
KNOWN_DISTANCE_METERS = 15.0

# Tracker settings
MAX_DISAPPEARED_FRAMES = 40          # frames before a lost vehicle is deregistered
MAX_MATCH_DISTANCE = 80              # max pixels for centroid matching

# Speed settings
SPEED_LIMIT_KMH = 60                 # speeds above this shown in RED
MIN_REASONABLE_SPEED = 5             # ignore speeds below this (likely noise)
MAX_REASONABLE_SPEED = 200           # ignore speeds above this (likely tracking error)

# Display settings
LINE_COLOR_1 = (0, 255, 255)         # Yellow for Line 1
LINE_COLOR_2 = (0, 0, 255)           # Red for Line 2
SAFE_SPEED_COLOR = (0, 255, 0)       # Green for normal speed
OVER_SPEED_COLOR = (0, 0, 255)       # Red for over speed limit
TEXT_COLOR = (255, 255, 255)         # White for text
```

**Include a comment explaining how to calibrate `KNOWN_DISTANCE_METERS`:**

```python
# HOW TO CALIBRATE:
# 1. Pause the video and identify a real-world reference between the two lines
#    (road markings, pole gaps, lane widths — Indian standard lane = 3.5m)
# 2. Count how many of those references fit between Line 1 and Line 2
# 3. Multiply: KNOWN_DISTANCE_METERS = count × reference_length
# 4. Example: 4 lane markings × 3.75m gap = 15.0 meters
#
# If you can't calibrate, use 15.0 as a reasonable default for a typical
# highway camera view — the speeds won't be exact but will be in the right range
```

#### SECTION 2: YOLO DETECTION FUNCTION

```python
def detect_vehicles(model, frame):
    """
    Run YOLOv8 on one frame and return bounding boxes of vehicles only.

    Args:
        model: loaded YOLOv8 model
        frame: BGR image (numpy array)

    Returns:
        list of (x1, y1, x2, y2, confidence, class_name) for each vehicle detected
    """
```

- Run `model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)`
- Filter results to only include VEHICLE_CLASSES
- Map class IDs to names: {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
- Return list of detections with bbox coordinates, confidence, and class name

#### SECTION 3: SPEED CALCULATION LOGIC

This is the core brain. Maintain these data structures:

```python
# Track when each vehicle crossed each line
line1_cross_time = {}    # {vehicle_id: timestamp_when_crossed_line1}
line2_cross_time = {}    # {vehicle_id: timestamp_when_crossed_line2}
vehicle_speeds = {}      # {vehicle_id: calculated_speed_kmh}
vehicle_classes = {}     # {vehicle_id: "Car" / "Bus" / "Truck" / "Motorcycle"}
crossed_line1 = set()    # vehicle IDs that have crossed Line 1
crossed_line2 = set()    # vehicle IDs that have crossed Line 2
vehicle_trail = defaultdict(lambda: deque(maxlen=30))  # last 30 centroids for trail drawing
```

**Line crossing detection logic (implement this carefully):**

```python
def check_line_crossing(vehicle_id, cy, current_time, frame_height):
    """
    Check if a vehicle's centroid cy has crossed Line 1 or Line 2.

    Logic:
    - Line 1 is at LINE_1_POSITION × frame_height
    - Line 2 is at LINE_2_POSITION × frame_height
    - A vehicle "crosses" a line when its centroid y-coordinate is within
      ±TOLERANCE pixels of the line position
    - We record the crossing time only ONCE per vehicle per line

    The tolerance band prevents missing a fast vehicle that jumps
    over the exact line pixel between frames.
    """
    TOLERANCE = 15  # pixels

    line1_y = int(frame_height * LINE_1_POSITION)
    line2_y = int(frame_height * LINE_2_POSITION)

    # Check Line 1 crossing
    if vehicle_id not in crossed_line1:
        if abs(cy - line1_y) < TOLERANCE:
            crossed_line1.add(vehicle_id)
            line1_cross_time[vehicle_id] = current_time

    # Check Line 2 crossing (only if already crossed Line 1)
    if vehicle_id in crossed_line1 and vehicle_id not in crossed_line2:
        if abs(cy - line2_y) < TOLERANCE:
            crossed_line2.add(vehicle_id)
            line2_cross_time[vehicle_id] = current_time

            # Calculate speed!
            time_diff = line2_cross_time[vehicle_id] - line1_cross_time[vehicle_id]
            if time_diff > 0:
                speed_ms = KNOWN_DISTANCE_METERS / time_diff
                speed_kmh = speed_ms * 3.6

                # Sanity check
                if MIN_REASONABLE_SPEED <= speed_kmh <= MAX_REASONABLE_SPEED:
                    vehicle_speeds[vehicle_id] = round(speed_kmh, 1)
```

#### SECTION 4: FRAME OVERLAY DRAWING

For each frame, draw these visual elements:

**a) The two virtual lines across the full frame width:**
```
Line 1 (Entry): Yellow dashed line with label "ENTRY LINE" on the left
Line 2 (Exit):  Red dashed line with label "EXIT LINE" on the left
```
Use `cv2.line()` with thickness 2. For dashed effect, draw short segments with gaps.

**b) For each tracked vehicle, draw:**
- A bounding box rectangle:
  - GREEN if speed is below SPEED_LIMIT_KMH or not yet calculated
  - RED if speed is above SPEED_LIMIT_KMH
- A label above the bounding box showing:
  - Vehicle ID and class: "ID:5 Car"
  - Speed if calculated: "47.3 km/h" in GREEN or "78.5 km/h ⚠" in RED
  - If between lines: "Measuring..." in YELLOW
- A motion trail: draw the last 30 centroid positions as small dots connected by a thin line, creating a "tail" behind the vehicle. Use the vehicle's color (green or red) with fading opacity.

**c) A dashboard overlay in the top-left corner (semi-transparent dark background):**
```
┌──────────────────────────────────────┐
│  🚗 Multi-Vehicle Speed Detector     │
│  Vehicles tracked: 12                │
│  Currently in zone: 4               │
│  Speeds captured: 8                 │
│  Avg speed: 45.2 km/h              │
│  Max speed: 72.1 km/h              │
│  Over speed limit: 2               │
│  FPS: 28.3                         │
└──────────────────────────────────────┘
```

Use `cv2.rectangle()` with `cv2.FILLED` and alpha blending for the semi-transparent background. Draw text with `cv2.putText()`.

**d) A vehicle count bar on the right side:**

Show a small colored bar for each vehicle type detected:
```
Cars: 6  🟩🟩🟩🟩🟩🟩
Trucks: 2  🟦🟦
Buses: 1  🟨
Bikes: 3  🟪🟪🟪
```

Implement using small filled rectangles.

#### SECTION 5: SPEED LOG CSV

Save every speed measurement to a CSV file in the `logs/` folder:

```csv
Timestamp,Vehicle_ID,Vehicle_Type,Speed_KMH,Over_Limit
2026-07-27 10:15:23,5,Car,47.3,No
2026-07-27 10:15:25,8,Truck,72.1,Yes
2026-07-27 10:15:28,11,Motorcycle,38.9,No
```

Write a row every time a vehicle's speed is calculated (when it crosses Line 2).

#### SECTION 6: VIDEO OUTPUT

Save the processed video (with all overlays) to the `output/` folder using `cv2.VideoWriter`. Match the input video's FPS and resolution. Codec should be `mp4v` for .mp4 output.

#### SECTION 7: INTERACTIVE CONTROLS

While the video plays, support these keyboard controls:

| Key | Action |
|-----|--------|
| `Q` | Quit and save everything |
| `SPACE` | Pause / Resume the video |
| `S` | Screenshot current frame → save to `output/screenshot_timestamp.jpg` |
| `+` / `-` | Increase / Decrease playback speed |
| `1` | Move Line 1 up by 10 pixels (for live calibration) |
| `2` | Move Line 1 down by 10 pixels |
| `3` | Move Line 2 up by 10 pixels |
| `4` | Move Line 2 down by 10 pixels |

The line adjustment keys are critical — at the exhibition, you can tune the line positions on the fly to match any video the judges throw at you.

#### SECTION 8: MAIN FUNCTION AND CLI

```python
def main():
    parser = argparse.ArgumentParser(description="Multi-Vehicle Speed Detection System")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default=None, help="Path to output video (optional)")
    parser.add_argument("--show", action="store_true", default=True, help="Show live preview window")
    parser.add_argument("--distance", type=float, default=KNOWN_DISTANCE_METERS,
                        help="Real-world distance between the two lines in meters")
    parser.add_argument("--speed-limit", type=float, default=SPEED_LIMIT_KMH,
                        help="Speed limit in km/h (speeds above this shown in red)")
    args = parser.parse_args()
```

**Startup sequence:**
1. Print a nice banner with the project name
2. Load YOLOv8n model (print "Loading YOLOv8n model..." → "Model loaded ✅")
3. Open the input video (validate it exists and can be read)
4. Print video info: resolution, FPS, total frames, duration
5. Initialize the centroid tracker
6. Start the main processing loop
7. On exit, print a summary:

```
╔══════════════════════════════════════╗
║         SESSION SUMMARY              ║
╠══════════════════════════════════════╣
║  Total vehicles tracked:  34         ║
║  Speeds captured:         28         ║
║  Average speed:           48.2 km/h  ║
║  Maximum speed:           78.5 km/h  ║
║  Minimum speed:           12.3 km/h  ║
║  Over speed limit:        5          ║
║  Video saved:  output/result.mp4     ║
║  Log saved:    logs/speed_log.csv    ║
╚══════════════════════════════════════╝
```

---

### ERROR HANDLING (Handle all of these)

- Video file not found → print "Error: Video file 'xyz.mp4' not found. Check the path." and exit
- Video can't be opened by OpenCV → print "Error: Cannot read video file. It may be corrupted or in an unsupported format." and exit
- YOLOv8 model download fails → catch the exception and print "Error: Cannot download YOLOv8n model. Check your internet connection. The model will be auto-downloaded on first run (~6MB)."
- No vehicles detected in any frame → don't crash, just show the video with the lines and dashboard saying "Vehicles tracked: 0"
- Video ends → gracefully close everything, print the summary, save the CSV
- User presses Q mid-video → same graceful shutdown
- Output video codec not supported → fall back to `XVID` codec with `.avi` extension
- Frame read fails mid-video → skip that frame, continue processing

---

### CODE STRUCTURE

Organize `speed_detector.py` in this order:

```python
# ===== IMPORTS =====

# ===== CONFIGURATION (all constants at the top) =====

# ===== HELPER FUNCTIONS =====
def detect_vehicles(model, frame):
    """Run YOLO and return vehicle bounding boxes."""

def check_line_crossing(vehicle_id, cy, current_time, frame_height):
    """Check and record line crossings, calculate speed."""

def draw_dashed_line(frame, pt1, pt2, color, thickness, gap):
    """Draw a dashed line on the frame."""

def draw_label(frame, text, x, y, bg_color, text_color):
    """Draw text with background rectangle for readability."""

def draw_dashboard(frame, stats):
    """Draw the semi-transparent stats overlay."""

def draw_vehicle_overlay(frame, vehicle_id, bbox, speed, vehicle_class, trail):
    """Draw bounding box, label, speed, and motion trail for one vehicle."""

def save_speed_log(vehicle_id, vehicle_class, speed, log_file):
    """Append one speed measurement to the CSV log."""

# ===== MAIN PROCESSING LOOP =====
def process_video(args):
    """Main function that processes the video frame by frame."""

# ===== CLI ENTRY POINT =====
def main():
    """Parse arguments and start processing."""

if __name__ == "__main__":
    main()
```

---

### SAMPLE VIDEOS FOR TESTING

The system should work with any traffic video from a fixed camera angle. For testing and the exhibition, use:

1. **Your own BengaluruTrafficAI sample video** — `ai/dataset/sample_traffic.mp4` from the reference repo
2. **Free traffic videos from Pexels** — search "traffic aerial" or "traffic highway CCTV" on pexels.com (all free for any use)
3. **Record your own** — mount a phone on a bridge or overpass overlooking a road, record for 2 minutes. This is the most impressive for the exhibition because you can calibrate the distance exactly.

---

### EXTRA FEATURES (Add these if time permits — do the basic version first)

1. **Vehicle count by direction:** If vehicles move both up and down in the video, detect direction from the centroid trail (if cy is increasing over frames → moving down, if decreasing → moving up). Show separate counts for each direction.

2. **Speed histogram:** After processing the full video, use matplotlib to generate a histogram of all captured speeds and save it as `output/speed_histogram.png`. Show this on the exhibition poster.

3. **Heatmap overlay:** Track where vehicles spend the most time (accumulate centroid positions) and overlay a semi-transparent heatmap showing high-traffic zones.

4. **Overspeed alert with snapshot:** When a vehicle exceeds the speed limit, automatically save a cropped image of that vehicle to `output/overspeeding/vehicle_ID_speed.jpg`. This simulates a real traffic camera challan system.

5. **Sound alert:** Play a beep when a vehicle exceeds the speed limit. Use `winsound.Beep(1000, 200)` on Windows.

6. **Multiple lane detection:** Divide the road into lanes (left, center, right) and report per-lane average speeds.

---

### TESTING CHECKLIST

After building, test these scenarios:

- [ ] Run on a traffic video — do bounding boxes appear on vehicles?
- [ ] Do vehicle IDs stay consistent as they move? (Same car keeps same ID number)
- [ ] Do vehicles crossing Line 1 get recorded? (Check terminal output)
- [ ] Do vehicles crossing Line 2 trigger a speed calculation?
- [ ] Are speeds in a reasonable range (10-120 km/h for city traffic)?
- [ ] Are over-speed vehicles shown in RED?
- [ ] Does the dashboard update in real-time?
- [ ] Does pressing Q exit gracefully and show the summary?
- [ ] Does SPACE pause/resume correctly?
- [ ] Can you adjust lines with keys 1/2/3/4 during playback?
- [ ] Is the output video saved with overlays?
- [ ] Is the CSV log saved with all speed measurements?
- [ ] What happens with no vehicles in frame? (Should not crash)
- [ ] What happens with overlapping vehicles? (Tracker should try to keep IDs separate)
- [ ] Does it work on a different video? (Test with at least 2 videos)

---

### EXHIBITION TIPS

1. **Pre-process a demo video** — before the exhibition, run the system on a good traffic clip and save the output. If YOLO is slow on the exhibition laptop, you can play the pre-processed video instead.

2. **Calibration story** — on your poster, explain HOW you calibrated the distance. "We used Google Maps satellite view to measure the distance between two road markings visible in our video. The distance was 15.2 meters." This shows judges you understand the engineering, not just the code.

3. **Accuracy disclaimer** — be honest: "Our system estimates speed with ±10-15% accuracy depending on camera angle and calibration. Professional speed cameras use radar/LIDAR for ±1% accuracy. Our approach is vision-only and intended as a proof of concept." Judges respect honesty more than false claims.

4. **Live demo flow:** Open the system → load a video → point at the lines and explain what they are → show a vehicle crossing both lines → its speed appears → "this vehicle is going 72 km/h, which is above our 60 km/h limit, so it's highlighted in red." → show the CSV log → show the speed histogram. 60 seconds, complete.

---

### WHAT THIS SHOULD NOT BE

- Do NOT use any web framework (no Flask, no React, no Streamlit) — this is a pure Python + OpenCV desktop application
- Do NOT use external tracking libraries (no SORT, no DeepSORT, no ByteTrack) — build the simple centroid tracker from scratch as described. That's the learning.
- Do NOT use any database — just CSV files
- Do NOT train a custom YOLO model — use the pre-trained `yolov8n.pt` which already detects cars, trucks, buses, motorcycles out of the box
- Do NOT hardcode any file paths — use `os.path.join()` and argparse everywhere
- Keep it to exactly TWO files: `speed_detector.py` and `tracker.py`

---

### FINAL OUTPUT

Give me two complete files:

1. **`tracker.py`** — the CentroidTracker class, ~60–80 lines, fully commented
2. **`speed_detector.py`** — the main program, ~200–250 lines, fully commented

Both should be fully working. When I run `python speed_detector.py --video traffic.mp4`, it should open the video, detect vehicles, track them, calculate speeds, show the live preview with all overlays, and save the output video and CSV log.

Also generate a `requirements.txt` and a short `README.md` with setup instructions.

---

## END OF PROMPT — STOP COPYING HERE

---

## How to Use This Prompt

1. Copy everything between "START COPYING FROM HERE" and "STOP COPYING HERE"
2. Paste it into a new Claude chat (or ChatGPT, or any AI coding tool)
3. It will generate `speed_detector.py`, `tracker.py`, `requirements.txt`, and `README.md`
4. Save the files, install the libraries, download a sample traffic video, and run it

## Installation Commands

```bash
# Install all required libraries
pip install ultralytics opencv-python numpy

# YOLOv8n model (~6MB) auto-downloads on first run — needs internet once

# Run the system
python speed_detector.py --video videos/sample_traffic.mp4

# With custom settings
python speed_detector.py --video traffic.mp4 --distance 12.0 --speed-limit 40 --output result.mp4
```

## Where to Get Sample Traffic Videos

- **Pexels (free):** Search "traffic highway" or "traffic CCTV" at pexels.com
- **Your previous project:** Use `ai/dataset/sample_traffic.mp4` from BengaluruTrafficAI
- **Record your own:** Mount a phone on a bridge or overpass for 2 minutes

## Common Problems

| Problem | Fix |
|---------|-----|
| YOLO model download fails | Check internet. Or manually download `yolov8n.pt` from ultralytics.com and place in the project folder |
| Very slow FPS | Use `yolov8n.pt` (nano), not `yolov8s.pt`. Or process at half resolution: add `frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)` |
| Speeds are wildly wrong | Calibrate `KNOWN_DISTANCE_METERS` — this is the most important setting. Use Google Maps to measure real road distance |
| Vehicle IDs keep changing | Increase `MAX_DISAPPEARED_FRAMES` to 50 or 60. Or decrease `MAX_MATCH_DISTANCE` if vehicles are getting swapped |
| No vehicles detected | Lower `CONFIDENCE_THRESHOLD` from 0.4 to 0.25. Or check if the video actually has vehicles visible |
| Lines are in wrong position | Use keys 1/2/3/4 during playback to adjust, or change `LINE_1_POSITION` and `LINE_2_POSITION` in config |
| Output video is 0 bytes | OpenCV codec issue — try changing `mp4v` to `XVID` and output to `.avi` |
