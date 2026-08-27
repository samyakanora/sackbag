Count sacks crossing a line

1. Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

2. Run the script (example):

```bash
python scripts/count_line_cross.py \
  --source runs/detect/predict-2/IMG_1047.mp4 \
  --weights yolo11n.pt \
  --class_name Sack \
  --line 0.5,0.0,0.5,1.0 \
  --output runs/detect/predict-2/IMG_1047_counted.mp4
```

3. Customize:
- Change `--line` to specify the crossing line in normalized coords `x1,y1,x2,y2`.
- Use `--class_id` if you prefer to target a numeric class index.
- Tune `--conf` and `--max-distance` for detection/tracking sensitivity.

RTSP live camera and phone updates

1. Create a Telegram bot and get `BOT_TOKEN` (BotFather) and your `CHAT_ID`.

2. Run the RTSP service to receive periodic updates on your phone:

```bash
python scripts/rtsp_count_server.py \
  --rtsp rtsp://user:pass@camera/live \
  --weights yolo11n.pt \
  --telegram-token BOT_TOKEN \
  --telegram-chat-id CHAT_ID \
  --line 0.5,0.9,0.5,0.1 \
  --update-interval 30
```

The service will send a message every `--update-interval` seconds with how many sacks were unloaded in that interval and the running total. It attaches an annotated snapshot when possible.
