Project: Sack Counting (Ultralytics YOLO + ByteTrack)

Overview
-
This repository contains an offline and RTSP-capable sack-counting application that uses an Ultralytics YOLO model and ByteTrack (via Ultralytics `model.track`) to assign persistent IDs. The counting logic is fixed and counts a `Sack` when any part of the detection bounding box touches/crosses a scaled counting line.

Important (frozen behavior)
-
- Model weights: `runs/detect/train/weights/best.pt` (do not modify or retrain)
- Offline confidence: `0.40` (set in `config.py` as `CONF_OFFLINE`)
- Counting: only the `Sack` class is counted (class name or `CLASS_ID` can be set in `config.py`)
- Counting uses full bounding-box vs line intersection (no centroid counting)
- ByteTrack is used to deduplicate counts by track ID

Quickstart (clean environment)
-
1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify Python syntax/imports (optional):

```bash
python -m py_compile $(find . -name '*.py')
# or
python -m compileall -q .
```

4. Run the offline counting test (preserves current logic):

```bash
python3 app.py offline --source "20260827170855041.MP4"
```

Notes
-
- Do NOT commit or change `runs/detect/train/weights/best.pt`.
- Secrets (RTSP credentials, Telegram tokens) must not be committed. See `.env.example` for environment variables.
- The `app.py` launcher wraps the existing scripts. The primary offline script is `scripts/count_line_cross.py`.

Files you will use most
-
- [app.py](app.py#L1) — launcher for `offline` and `rtsp` modes
- [config.py](config.py#L1) — central defaults and environment overrides (CONF_OFFLINE = 0.40)
- [scripts/count_line_cross.py](scripts/count_line_cross.py#L1) — offline counting implementation (ByteTrack)
- [scripts/rtsp_count_server.py](scripts/rtsp_count_server.py#L1) — RTSP service (do not enable without secrets)
- [storage.py](storage.py#L1) — lightweight sqlite persistence (optional)

Support
-
If you want me to (after you approve these changes):
- run an automated extraction of verification frames per counted event
- prepare a small test harness to run on a fresh machine and validate end-to-end

License / Privacy
-
This repository may contain captured video or verification frames in `runs/` — these are ignored by `.gitignore` by default. Ensure you do not commit sensitive files.
