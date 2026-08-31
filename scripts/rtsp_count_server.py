#!/usr/bin/env python3
"""
Lightweight RTSP counting service: reads RTSP, counts class crossings, sends Telegram updates.

Usage example:
  python scripts/rtsp_count_server.py \
    --rtsp rtsp://user:pass@camera/live \
    --weights yolo11n.pt \
    --telegram-token YOUR_TOKEN \
    --telegram-chat-id YOUR_CHAT_ID \
    --line 0.5,0.9,0.5,0.1 \
    --update-interval 30

Requires: `pip install ultralytics opencv-python numpy requests`
"""
import argparse
import time
import os
import math
import sys
import tempfile
from collections import OrderedDict

import cv2
import numpy as np
import requests
from datetime import datetime

try:
    from storage import init_db, get_today_total, increment, log_event
    init_db()
except Exception:
    def get_today_total():
        return 0
    def increment(amount=1):
        return None
    def log_event(*args, **kwargs):
        return None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=60):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.last_box = {}
        self.disappeared = OrderedDict()
        self.counted = {}
        self.max_disappeared = max_disappeared * 2
        self.max_distance = max_distance
        self.min_iou = 0.3

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.last_box[self.next_object_id] = None
        self.disappeared[self.next_object_id] = 0
        self.counted[self.next_object_id] = False
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.last_box:
            del self.last_box[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.counted:
            del self.counted[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # simple distance matching (kept for compatibility)
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])) - used_rows
        unused_cols = set(range(0, D.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects


def parse_line(arg):
    parts = [float(x) for x in arg.split(',')]
    if len(parts) != 4:
        raise ValueError('--line expects 4 comma-separated floats: x1,y1,x2,y2')
    return parts


def _on_segment(p, q, r):
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def _orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def _segments_intersect(p1, p2, q1, q2):
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, q1, p2):
        return True
    if o2 == 0 and _on_segment(p1, q2, p2):
        return True
    if o3 == 0 and _on_segment(q1, p1, q2):
        return True
    if o4 == 0 and _on_segment(q1, p2, q2):
        return True
    return False


def box_intersects_line(box, a, b):
    x1, y1, x2, y2 = box
    rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for i in range(4):
        if _segments_intersect(a, b, rect[i], rect[(i + 1) % 4]):
            return True
    if x1 <= a[0] <= x2 and y1 <= a[1] <= y2:
        return True
    if x1 <= b[0] <= x2 and y1 <= b[1] <= y2:
        return True
    return False


def send_telegram(token, chat_id, text, photo_path=None):
    base = f'https://api.telegram.org/bot{token}'
    try:
        if photo_path and os.path.exists(photo_path):
            files = {'photo': open(photo_path, 'rb')}
            data = {'chat_id': chat_id, 'caption': text}
            resp = requests.post(base + '/sendPhoto', files=files, data=data, timeout=20)
        else:
            data = {'chat_id': chat_id, 'text': text}
            resp = requests.post(base + '/sendMessage', data=data, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print('Telegram send error:', e)
        return False


def annotate_frame(frame, boxes, centroids, ids, a, b, total_count, interval_count):
    out = frame.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for obj_id, centroid in ids.items():
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.circle(out, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(out, f'ID {obj_id}', (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(out, a, b, (0, 0, 255), 2)
    cv2.putText(out, f'Total: {total_count}  Interval: {interval_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    try:
        import config
    except Exception:
        config = None

    parser.add_argument('--rtsp', default=(config.RTSP_URL if config else ''), help='RTSP URL')
    parser.add_argument('--weights', default=(config.WEIGHTS if config else 'yolo11n.pt'))
    parser.add_argument('--line', default=(config.LINE_RTSP if config else '0.5,0.9,0.5,0.1'))
    parser.add_argument('--conf', type=float, default=(config.CONF_RTSP if config else 0.35))
    parser.add_argument('--class_name', default=(config.CLASS_NAME if config else 'Sack'))
    parser.add_argument('--class_id', type=int, default=(config.CLASS_ID if config else None))
    parser.add_argument('--telegram-token', default=(config.TELEGRAM_TOKEN if config else ''), help='Telegram bot token')
    parser.add_argument('--telegram-chat-id', default=(config.TELEGRAM_CHAT_ID if config else ''), help='Telegram chat id')
    parser.add_argument('--update-interval', type=int, default=(config.UPDATE_INTERVAL if config else 30), help='seconds between phone updates')
    parser.add_argument('--max-distance', type=int, default=(config.MAX_DISTANCE if config else 60))
    parser.add_argument('--save-snapshots', action='store_true', default=(config.SAVE_SNAPSHOTS if config else False))
    args = parser.parse_args()

    # Print configured RTSP source but hide the password.
    def _mask_rtsp(url):
        if not url:
            return '(empty)'
        try:
            import re
            m = re.match(r'(rtsp://[^:]+:)[^@]+(@.+)', url)
            if m:
                return m.group(1) + '*****' + m.group(2)
        except Exception:
            pass
        return '(hidden)'

    print('RTSP source:', _mask_rtsp(args.rtsp))

    if YOLO is None:
        print('Please install ultralytics: pip install ultralytics')
        sys.exit(1)

    model = YOLO(args.weights)

    cap = None
    retry_delay = 5
    while True:
        try:
            cap = cv2.VideoCapture(args.rtsp)
            if cap.isOpened():
                break
            else:
                print('Failed to open RTSP, retrying...')
                time.sleep(retry_delay)
        except Exception as e:
            print('Error opening RTSP:', e)
            time.sleep(retry_delay)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # Determine counting line coordinates. Support both normalized (0..1)
    # and reference-pixel coords (scale from config.REFERENCE_WIDTH/HEIGHT).
    try:
        ref_w = getattr(config, 'REFERENCE_WIDTH', 582) if config else 582
        ref_h = getattr(config, 'REFERENCE_HEIGHT', 328) if config else 328
    except Exception:
        ref_w, ref_h = 582, 328

    lx1, ly1, lx2, ly2 = parse_line(args.line)
    if 0.0 <= lx1 <= 1.0 and 0.0 <= ly1 <= 1.0 and 0.0 <= lx2 <= 1.0 and 0.0 <= ly2 <= 1.0:
        # normalized coordinates relative to frame size
        a = (int(lx1 * width), int(ly1 * height))
        b = (int(lx2 * width), int(ly2 * height))
    else:
        # treat as reference-pixel coords and scale to actual frame
        a = (int(lx1 * width / ref_w), int(ly1 * height / ref_h))
        b = (int(lx2 * width / ref_w), int(ly2 * height / ref_h))

    tracker = CentroidTracker(max_distance=args.max_distance)
    object_prev_side = {}
    # load today's existing total if storage is available
    total_count = get_today_total() or 0
    interval_count = 0

    last_update = time.time()

    snapshot_dir = tempfile.mkdtemp(prefix='rtsp_snap_') if args.save_snapshots else None

    print('Starting RTSP processing...')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Frame read failed, reconnecting...')
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(args.rtsp)
            time.sleep(1)
            continue

        try:
            results = model.predict(source=frame, imgsz=640, conf=args.conf, device='')
        except Exception:
            results = model(frame)

        r = results[0]
        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            try:
                xyxy = r.boxes.xyxy.numpy()
                confs = r.boxes.conf.numpy()
                clss = r.boxes.cls.numpy().astype(int)
            except Exception:
                xyxy = np.array([])
                confs = np.array([])
                clss = np.array([])

        # determine target class id
        if args.class_id is not None:
            target_cls = args.class_id
        else:
            try:
                names = model.names
                target_cls = None
                for k, v in names.items():
                    if v.lower() == args.class_name.lower():
                        target_cls = int(k)
                        break
                if target_cls is None:
                    target_cls = 1
            except Exception:
                target_cls = 1

        det_centroids = []
        det_boxes = []
        for i, box in enumerate(xyxy):
            if confs[i] < args.conf:
                continue
            if len(clss) > 0 and clss[i] != target_cls:
                continue
            x1, y1, x2, y2 = box.astype(int)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            det_centroids.append((cx, cy))
            det_boxes.append((x1, y1, x2, y2))

        objects = tracker.update(det_centroids)

        used = set()
        for obj_id, centroid in objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            best_idx = None
            best_dist = float('inf')
            for i, c in enumerate(det_centroids):
                if i in used:
                    continue
                d = math.hypot(cx - c[0], cy - c[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is not None and best_dist < args.max_distance:
                used.add(best_idx)
                x1, y1, x2, y2 = det_boxes[best_idx]
                # if any part of the detected box touches the line, count it once
                if box_intersects_line((x1, y1, x2, y2), a, b):
                    if not tracker.counted.get(obj_id, False):
                        total_count += 1
                        interval_count += 1
                        tracker.counted[obj_id] = True
                        # persist and log event
                        try:
                            increment(1)
                            log_event(obj_id=obj_id, source='rtsp', details=f'box={x1},{y1},{x2},{y2}')
                        except Exception:
                            pass
            object_prev_side[obj_id] = None

        now = time.time()
        if now - last_update >= args.update_interval:
            text = f'Sacks unloaded (last {args.update_interval}s): {interval_count} — Total: {total_count}'
            # annotate a snapshot
            try:
                annotated = annotate_frame(frame, det_boxes, objects, objects, a, b, total_count, interval_count)
                tmpf = None
                if snapshot_dir:
                    tmpf = os.path.join(snapshot_dir, f'snap_{int(now)}.jpg')
                    cv2.imwrite(tmpf, annotated)
                else:
                    tmp_fd, tmpf = tempfile.mkstemp(suffix='.jpg')
                    os.close(tmp_fd)
                    cv2.imwrite(tmpf, annotated)
                send_telegram(args.telegram_token, args.telegram_chat_id, text, photo_path=tmpf)
                if not args.save_snapshots and tmpf and os.path.exists(tmpf):
                    os.remove(tmpf)
            except Exception as e:
                print('Failed to send snapshot:', e)
                send_telegram(args.telegram_token, args.telegram_chat_id, text)
            interval_count = 0
            last_update = now

        # display annotated frame and allow quitting with 'q'
        try:
            # create annotated frame for display (reuse same annotation used for snapshots)
            annotated_frame = annotate_frame(frame, det_boxes, objects, objects, a, b, total_count, interval_count)
            cv2.imshow("Sack Counter - Live RTSP", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Quit key pressed, shutting down...')
                try:
                    cap.release()
                except Exception:
                    pass
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                sys.exit(0)
        except Exception:
            # If display fails (e.g., headless), continue without crashing
            pass

        # small sleep to be cooperative
        time.sleep(0.01)


if __name__ == '__main__':
    main()
