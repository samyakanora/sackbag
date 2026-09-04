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
    from storage import (
        init_db,
        get_today_total,
        increment,
        log_event,
        record_sack,
        get_sacks_last_hour,
        get_last_sack_time,
        get_recent_sack_events,
        record_interval,
        get_last_interval,
        get_sack_by_number,
    )
    init_db()
except Exception:
    def get_today_total():
        return 0
    def increment(amount=1):
        return None
    def log_event(*args, **kwargs):
        return None
    def record_sack(*args, **kwargs):
        return None
    def get_sacks_last_hour():
        return 0
    def get_last_sack_time():
        return None
    def get_recent_sack_events(limit=10):
        return []
    def record_interval(*args, **kwargs):
        return None
    def get_last_interval():
        return None
    def get_sack_by_number(n):
        return None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from scripts.recorder import Recorder
except Exception:
    try:
        from recorder import Recorder
    except Exception:
        Recorder = None


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


def _point_side(a, b, p):
    # returns signed side: >0 left, <0 right, 0 on line
    # compute cross product of AB and AP
    (x1, y1), (x2, y2) = a, b
    (px, py) = p
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _centroid_distance(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def _cleanup_recent(recent_list, cooldown_seconds):
    now = time.time()
    # remove entries older than cooldown * 2 to keep list small
    cutoff = now - (cooldown_seconds * 2)
    while recent_list and recent_list[0]['ts'] < cutoff:
        recent_list.pop(0)


def _is_duplicate_event(recent_list, centroid, bbox, ts, cooldown_seconds, distance_pixels):
    # check recent_list for events within cooldown_seconds and within distance_pixels
    for ev in recent_list[::-1]:
        if abs(ev['ts'] - ts) > cooldown_seconds:
            continue
        # centroid proximity
        if _centroid_distance(ev['centroid'], centroid) <= distance_pixels:
            return True
        # bbox proximity (center-based)
        ev_cx = int((ev['bbox'][0] + ev['bbox'][2]) / 2)
        ev_cy = int((ev['bbox'][1] + ev['bbox'][3]) / 2)
        if _centroid_distance((ev_cx, ev_cy), centroid) <= distance_pixels:
            return True
    return False


def _find_matching_physical(recent_phys, centroid, bbox, ts, side_val, cooldown_seconds, distance_pixels, max_recent=200):
    # conservative matching: require time + distance or size similarity
    best = None
    best_score = 0
    cx, cy = centroid
    area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    for phys in reversed(recent_phys[-max_recent:]):
        dt = abs(ts - phys.get('last_ts', 0))
        pcx, pcy = phys.get('last_centroid', (0, 0))
        dist = _centroid_distance((pcx, pcy), centroid)
        pbbox = phys.get('last_bbox', (0, 0, 0, 0))
        parea = max(1, (pbbox[2] - pbbox[0]) * (pbbox[3] - pbbox[1]))
        size_ratio = max(area, parea) / min(area, parea)

        score = 0
        if dt <= cooldown_seconds:
            score += 2
        if dist <= distance_pixels:
            score += 2
        if size_ratio <= 2.0:
            score += 1
        # same side increases confidence
        if phys.get('last_side') is not None and side_val is not None and (phys.get('last_side') * side_val) >= 0:
            score += 1

        if score > best_score:
            best_score = score
            best = phys

    # require minimum score to accept match (conservative)
    if best_score >= 3:
        return best
    return None


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def _bbox_center(box):
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))


def _associate_to_stable(stable_list, track_id, centroid, bbox, ts, side_val,
                         max_dist=80, min_iou=0.3, timeout=1.5, max_recent=200, debug=False):
    # quick exact-track update
    for s in reversed(stable_list[-max_recent:]):
        if s.get('current_track_id') == track_id:
            s['previous_track_ids'].add(track_id)
            s['current_track_id'] = track_id
            s['bbox'] = bbox
            s['center'] = centroid
            s['last_seen_ts'] = ts
            s['last_side'] = side_val
            hist = s.setdefault('history', [])
            hist.append({'center': centroid, 'ts': ts, 'bbox': bbox})
            if len(hist) > 5:
                hist.pop(0)
            if len(hist) >= 2:
                h0, h1 = hist[-2], hist[-1]
                dt_h = max(1e-3, h1['ts'] - h0['ts'])
                vx = (h1['center'][0] - h0['center'][0]) / dt_h
                vy = (h1['center'][1] - h0['center'][1]) / dt_h
                s['velocity'] = (vx, vy)
            else:
                s['velocity'] = s.get('velocity', (0.0, 0.0))
            return s, False, False

    best = None
    best_conf = 0.0
    for s in reversed(stable_list[-max_recent:]):
        dt = ts - s.get('last_seen_ts', 0)
        if dt < 0:
            dt = abs(dt)

        vx, vy = s.get('velocity', (0.0, 0.0))
        pred_cx = s.get('center', (0, 0))[0] + vx * dt
        pred_cy = s.get('center', (0, 0))[1] + vy * dt
        pred_center = (pred_cx, pred_cy)

        pred_dist = _centroid_distance(pred_center, centroid)
        iou = _iou(s.get('bbox', (0, 0, 0, 0)), bbox)
        sarea = max(1, (s.get('bbox', (0, 0, 0, 0))[2] - s.get('bbox', (0, 0, 0, 0))[0]) * (s.get('bbox', (0, 0, 0, 0))[3] - s.get('bbox', (0, 0, 0, 0))[1]))
        area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        size_ratio = min(sarea, area) / max(sarea, area)

        disp_x = centroid[0] - s.get('center', (0, 0))[0]
        disp_y = centroid[1] - s.get('center', (0, 0))[1]
        disp_norm = math.hypot(disp_x, disp_y)
        vel_norm = math.hypot(vx, vy)
        dir_score = 0.5
        if disp_norm > 1e-3 and vel_norm > 1e-3:
            dot = (disp_x * vx + disp_y * vy) / (disp_norm * vel_norm)
            dir_score = (dot + 1.0) / 2.0

        time_factor = 1.0
        if dt > max(timeout, 0.001):
            time_factor = max(0.0, 1.0 - ((dt - timeout) / (timeout * 5.0)))

        dist_score = max(0.0, 1.0 - (pred_dist / max(1.0, max_dist * 2)))

        w_pred, w_iou, w_size, w_dir, w_time = 0.35, 0.30, 0.15, 0.10, 0.10
        conf = (w_pred * dist_score + w_iou * iou + w_size * size_ratio + w_dir * dir_score + w_time * time_factor)

        if conf > best_conf:
            best_conf = conf
            best = {
                'stable': s,
                'pred_center': pred_center,
                'pred_dist': pred_dist,
                'iou': iou,
                'size_ratio': size_ratio,
                'dir_score': dir_score,
                'dt': dt,
                'conf': conf,
            }

    CONF_THRESHOLD = 0.45
    if best and best['conf'] >= CONF_THRESHOLD:
        s = best['stable']
        was_reassoc = track_id not in s.get('previous_track_ids', set()) and track_id != s.get('current_track_id')
        s['previous_track_ids'].add(s.get('current_track_id'))
        s['previous_track_ids'].add(track_id)
        s['current_track_id'] = track_id
        s['bbox'] = bbox
        s['center'] = centroid
        s['last_seen_ts'] = ts
        s['last_side'] = side_val
        hist = s.setdefault('history', [])
        hist.append({'center': centroid, 'ts': ts, 'bbox': bbox})
        if len(hist) > 5:
            hist.pop(0)
        if len(hist) >= 2:
            h0, h1 = hist[-2], hist[-1]
            dt_h = max(1e-3, h1['ts'] - h0['ts'])
            vx = (h1['center'][0] - h0['center'][0]) / dt_h
            vy = (h1['center'][1] - h0['center'][1]) / dt_h
            s['velocity'] = (vx, vy)
        else:
            s['velocity'] = s.get('velocity', (0.0, 0.0))

        s['_assoc_diag'] = {
            'pred_center': best['pred_center'],
            'actual_center': centroid,
            'pred_dist': best['pred_dist'],
            'iou': best['iou'],
            'size_ratio': best['size_ratio'],
            'dir_score': best['dir_score'],
            'dt': best['dt'],
            'conf': best['conf'],
        }

        return s, False, was_reassoc

    return None, True, False


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


def annotate_frame(frame, boxes, centroids, ids, a, b, total_count, interval_count, labels=None, statuses=None, cfg_debug=False, highlight_ids=None):
    out = frame.copy()
    highlight_ids = set() if highlight_ids is None else set(highlight_ids)
    # draw detection boxes with status-aware coloring
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = (0, 255, 0)  # accepted by default
        text = ''
        if labels and i < len(labels) and labels[i]:
            text = labels[i]
        if statuses and i < len(statuses) and statuses[i]:
            st = statuses[i]
            if st.startswith('person'):
                color = (255, 165, 0)
            elif st.startswith('filtered'):
                color = (0, 140, 255)
            elif st.startswith('rejected'):
                color = (0, 0, 255)
            elif st.startswith('accepted'):
                color = (0, 255, 0)
            text = (text + ' ' + st).strip()
        # draw a visible count highlight for recently counted objects
        box_id = None
        if i in range(len(boxes)):
            # best effort: if this detection corresponds to the tracked object order, highlight by track id if available
            pass
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if text and cfg_debug:
            cv2.putText(out, text, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # draw tracked object ids
    for obj_id, centroid in ids.items():
        cx, cy = int(centroid[0]), int(centroid[1])
        is_highlight = obj_id in highlight_ids
        marker_color = (0, 255, 255) if is_highlight else (255, 0, 0)
        cv2.circle(out, (cx, cy), 4, marker_color, -1)
        cv2.putText(out, f'ID {obj_id}', (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if is_highlight:
            cv2.putText(out, 'COUNTED!', (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # draw counting line
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
    parser.add_argument('--test-rtsp', action='store_true', help='Test RTSP connectivity and exit')
    parser.add_argument('--show-line', action='store_true', help='Show configured counting line on first RTSP frame and exit')
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

    # Resolve RTSP URL from CLI -> ENV -> config
    resolved_rtsp = args.rtsp if args.rtsp else os.getenv('RTSP_URL', '')
    try:
        if not resolved_rtsp and config:
            resolved_rtsp = getattr(config, 'RTSP_URL', '') or resolved_rtsp
    except Exception:
        pass

    # Update args.rtsp with resolved value
    args.rtsp = resolved_rtsp

    # Diagnostics
    from urllib.parse import urlparse
    if args.rtsp:
        try:
            parsed = urlparse(args.rtsp)
            host = parsed.hostname or '(unknown)'
        except Exception:
            host = '(parse-failed)'
        print('RTSP URL configured: yes')
        print('RTSP URL host:', host)
        print('RTSP source:', _mask_rtsp(args.rtsp))

        # Validate parsed URL: must have a hostname and not contain placeholder USER/PASS
        invalid = False
        reason = ''
        try:
            if not parsed.hostname:
                invalid = True
                reason = 'missing hostname in RTSP URL'
            if parsed.username and parsed.username.upper() in ('USER', 'USERNAME'):
                invalid = True
                reason = 'RTSP username appears to be a placeholder'
            if parsed.password and parsed.password.upper() in ('PASS', 'PASSWORD'):
                invalid = True
                reason = 'RTSP password appears to be a placeholder'
        except Exception:
            invalid = True
            reason = 'invalid RTSP URL format'

        if invalid:
            print(f'Invalid RTSP URL: {reason}. Sanitized: {_mask_rtsp(args.rtsp)} Host: {host}')
            print('Please set RTSP_URL environment variable or pass --rtsp with a valid URL (username/password must be provided or omitted).')
            sys.exit(1)
    else:
        print('RTSP_URL is empty. Set RTSP_URL before starting the server.')
        sys.exit(1)

    # If user requested only to test RTSP connectivity or show the counting line, perform those actions and exit.
    if args.test_rtsp or args.show_line:
        cap_test = None
        try:
            cap_test = cv2.VideoCapture(args.rtsp)
            if not cap_test.isOpened():
                print('RTSP test: failed to open stream. Sanitized URL:', _mask_rtsp(args.rtsp))
                if cap_test is not None:
                    cap_test.release()
                sys.exit(1)
            ok, frame = cap_test.read()
            if not ok or frame is None:
                print('RTSP test: opened stream but failed to read a frame. Sanitized URL:', _mask_rtsp(args.rtsp))
                cap_test.release()
                sys.exit(1)

            print('RTSP test: stream opened and first frame read successfully. Host:', host)

            if args.show_line:
                # compute scaled line endpoints using same logic as main loop
                try:
                    ref_w = getattr(config, 'REFERENCE_WIDTH', 582) if config else 582
                    ref_h = getattr(config, 'REFERENCE_HEIGHT', 328) if config else 328
                except Exception:
                    ref_w, ref_h = 582, 328
                h, w = frame.shape[:2]
                try:
                    lx1, ly1, lx2, ly2 = parse_line(args.line)
                except Exception:
                    print('Failed to parse configured line:', args.line)
                    cap_test.release()
                    sys.exit(1)
                if 0.0 <= lx1 <= 1.0 and 0.0 <= ly1 <= 1.0 and 0.0 <= lx2 <= 1.0 and 0.0 <= ly2 <= 1.0:
                    a = (int(lx1 * w), int(ly1 * h))
                    b = (int(lx2 * w), int(ly2 * h))
                else:
                    a = (int(lx1 * w / ref_w), int(ly1 * h / ref_h))
                    b = (int(lx2 * w / ref_w), int(ly2 * h / ref_h))

                # draw and show/save
                try:
                    os.makedirs(os.path.join('runs', 'detect', 'verify'), exist_ok=True)
                    preview = frame.copy()
                    cv2.line(preview, a, b, (0, 0, 255), 3)
                    path = os.path.join('runs', 'detect', 'verify', 'rtsp_line_check.jpg')
                    cv2.imwrite(path, preview)
                    print('Saved RTSP first-frame line check image to', path)
                    try:
                        cv2.imshow('RTSP Line Check', preview)
                        print('Press any key on the image window to continue...')
                        cv2.waitKey(0)
                        cv2.destroyWindow('RTSP Line Check')
                    except Exception:
                        # headless environment; skip interactive display
                        pass
                except Exception as e:
                    print('Failed to create/show line-check image:', e)

            cap_test.release()
            sys.exit(0)
        except SystemExit:
            raise
        except Exception as e:
            try:
                if cap_test is not None:
                    cap_test.release()
            except Exception:
                pass
            print('RTSP test error:', e)
            sys.exit(1)

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
                print('[1] RTSP CAPTURE OPENED')
                break
            else:
                print('Failed to open RTSP, retrying...')
                time.sleep(retry_delay)
        except Exception as e:
            print('Error opening RTSP:', e)
            time.sleep(retry_delay)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # Simple direct-pixel RTSP line. No second scaling.
    try:
        line_parts = [float(x) for x in str(args.line).split(',')]
        if len(line_parts) != 4:
            raise ValueError('invalid RTSP line format')
        a = (int(line_parts[0]), int(line_parts[1]))
        b = (int(line_parts[2]), int(line_parts[3]))
    except Exception:
        a = (870, 1400)
        b = (2340, 0)

    try:
        line_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if line_len <= 0:
            line_len = 1.0
    except Exception:
        line_len = 1.0

    # Startup info
    print('RTSP CONNECTED')
    print(f'Resolution: {width}x{height}')
    print(f'FRAME SIZE: {width}x{height}')
    print(f'Counting line: ({a[0]},{a[1]}) -> ({b[0]},{b[1]})')
    print('Starting live sack processing...')

    # Confirm GUI support before trying to open a window.
    gui_ok = True
    try:
        info = cv2.getBuildInformation()
        print('OpenCV GUI backend info:')
        for line in info.splitlines():
            if 'GUI:' in line or 'HIGHGUI' in line or 'Video I/O:' in line:
                print(line)
        if 'GUI:' in info and 'COCOA' not in info and 'GTK' not in info and 'Qt' not in info:
            gui_ok = False
    except Exception:
        gui_ok = True
    if not gui_ok:
        print('OpenCV GUI is not available. Install opencv-python (GUI-enabled), not opencv-python-headless.')
        cap.release()
        sys.exit(1)

    # Save a first-frame verification image showing the configured counting line
    try:
        ok, first = cap.read()
        if ok and first is not None:
            try:
                os.makedirs(os.path.join('runs', 'detect', 'verify'), exist_ok=True)
                frame_check = first.copy()
                cv2.line(frame_check, a, b, (0, 0, 255), 3)
                check_path = os.path.join('runs', 'detect', 'verify', 'rtsp_line_check.jpg')
                cv2.imwrite(check_path, frame_check)
                print('Saved RTSP first-frame line check image to', check_path)
            except Exception as e:
                print('Failed to save RTSP line check image:', e)
        else:
            print('Warning: could not read first RTSP frame for line check')
    except Exception as e:
        print('RTSP line-check capture failed:', e)

    try:
        max_disp = getattr(config, 'TRACK_MAX_DISAPPEARED', 30) if config else 30
    except Exception:
        max_disp = 30
    tracker = CentroidTracker(max_disappeared=max_disp, max_distance=args.max_distance)
    object_prev_side = {}
    recent_events = []
    stable_objects = []
    _stable_counter = {'v': 1}
    assoc_count = 0
    # load stable-association config
    try:
        import config as _cfg
        cfg_track_dist = getattr(_cfg, 'TRACK_REASSOCIATION_DISTANCE', 80)
        cfg_track_iou = getattr(_cfg, 'TRACK_REASSOCIATION_IOU', 0.3)
        cfg_track_timeout = float(getattr(_cfg, 'TRACK_REASSOCIATION_TIMEOUT', 1.5))
        cfg_min_w = int(getattr(_cfg, 'MIN_PRODUCT_WIDTH', 20))
        cfg_min_h = int(getattr(_cfg, 'MIN_PRODUCT_HEIGHT', 20))
        cfg_min_area = int(getattr(_cfg, 'MIN_PRODUCT_AREA', 400))
        cfg_debug = bool(getattr(_cfg, 'DEBUG_TRACKING', False))
    except Exception:
        cfg_track_dist = 80
        cfg_track_iou = 0.3
        cfg_track_timeout = 1.5
        cfg_min_w = 20
        cfg_min_h = 20
        cfg_min_area = 400
        cfg_debug = False
    # load today's existing total if storage is available
    total_count = get_today_total() or 0
    interval_count = 0

    last_update = time.time()

    snapshot_dir = tempfile.mkdtemp(prefix='rtsp_snap_') if args.save_snapshots else None

    # initialize recorder if enabled
    recorder = None
    try:
        import config as _cfg
        if getattr(_cfg, 'RECORDING_ENABLED', False) and Recorder is not None:
            recorder = Recorder(directory=getattr(_cfg, 'RECORDING_DIRECTORY', 'recordings'),
                                segment_minutes=getattr(_cfg, 'RECORDING_SEGMENT_MINUTES', 10),
                                retention_days=getattr(_cfg, 'RECORDING_RETENTION_DAYS', 1))
    except Exception:
        recorder = None

    # monitoring state
    last_sack_time = None
    idle_timeout = getattr(config, 'IDLE_TIMEOUT', 60) if config else 60
    milestone = getattr(config, 'SACK_MILESTONE', 35) if config else 35

    print('Starting RTSP processing...')
    stop_requested = False
    last_annotated = None
    frames_processed = 0
    live_window_ready = False
    count_flash = {}
    first_frame_logged = False
    try:
        cv2.namedWindow('Sack Counter - Live', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sack Counter - Live', 1280, 720)
        while not stop_requested:
            if cap is None or not cap.isOpened():
                print('RTSP RECONNECTING...')
                try:
                    cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(args.rtsp)
                if not cap.isOpened():
                    print('RTSP reconnect failed; retrying in 5 seconds...')
                    time.sleep(5)
                    continue
                print('RTSP CONNECTED')

            ret, frame = cap.read()
            if not ret:
                print('RTSP FRAME READ FAILED')
                print(f'Reason: frame read returned false after {frames_processed} successfully processed frames. Reconnecting RTSP stream...')
                try:
                    cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(args.rtsp)
                if not cap.isOpened():
                    print('RTSP reconnect failed; waiting before retry...')
                    time.sleep(5)
                    continue
                print('RTSP RECONNECTED')
                continue

            frames_processed += 1
            if frames_processed == 1:
                print('FIRST FRAME RECEIVED')

            print('FIRST YOLO FRAME PROCESSED') if frames_processed == 1 else None
            try:
                results = model.predict(source=frame, imgsz=640, conf=args.conf, device='')
            except Exception:
                results = model(frame)
            if frames_processed == 1:
                print('YOLO COMPLETE')

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

            # collect detections; separate person boxes for overlap checks
            det_centroids = []
            det_boxes = []
            det_labels = []
            det_statuses = []
            person_boxes = []

            for i, box in enumerate(xyxy):
                if confs[i] < args.conf:
                    continue
                cls_i = int(clss[i]) if len(clss) > 0 else None
                cls_name = None
                try:
                    cls_name = model.names.get(cls_i, str(cls_i)).lower() if model is not None and hasattr(model, 'names') else str(cls_i)
                except Exception:
                    cls_name = str(cls_i)
                x1, y1, x2, y2 = box.astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                det_centroids.append((cx, cy))
                det_boxes.append((x1, y1, x2, y2))
                det_labels.append(f'{cls_name}:{confs[i]:.2f}')
                det_statuses.append('candidate')
                if cls_name == 'person' or (args.class_id is not None and cls_i is not None and cls_i != target_cls and cls_name == 'person'):
                    person_boxes.append((x1, y1, x2, y2))
                    det_statuses[-1] = 'person'

            track_centroids = []
            track_map = []
            for idx, (cxcy, box, label, status) in enumerate(zip(det_centroids, det_boxes, det_labels, det_statuses)):
                x1, y1, x2, y2 = box
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                area = w * h
                if w < cfg_min_w or h < cfg_min_h or area < cfg_min_area:
                    det_statuses[idx] = 'filtered_size'
                    if cfg_debug:
                        print('FILTER: small detection', box, 'w,h,area=', w, h, area)
                    continue
                aspect = float(w) / float(h) if h > 0 else 0.0
                if aspect < getattr(config, 'MIN_PRODUCT_ASPECT_MIN', 0.4) or aspect > getattr(config, 'MIN_PRODUCT_ASPECT_MAX', 2.5):
                    det_statuses[idx] = 'filtered_aspect'
                    if cfg_debug:
                        print('FILTER: aspect out of range', aspect, 'box=', box)
                    continue
                max_iou = 0.0
                max_person_area = 0
                for pb in person_boxes:
                    iou_val = _iou(pb, box)
                    if iou_val > max_iou:
                        max_iou = iou_val
                        max_person_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
                if max_iou >= getattr(config, 'PERSON_OVERLAP_THRESHOLD', 0.5):
                    if max_person_area >= (0.6 * area) or max_iou > 0.7:
                        det_statuses[idx] = 'filtered_person_overlap'
                        if cfg_debug:
                            print('FILTER: person overlap', box, 'max_iou=', max_iou)
                        continue
                det_statuses[idx] = 'accepted'
                track_centroids.append(cxcy)
                track_map.append(idx)

            objects = tracker.update(track_centroids)
            if frames_processed == 1:
                print('FIRST TRACKED FRAME PROCESSED')
                print('TRACKING COMPLETE')

            used = set()
            for obj_id, centroid in objects.items():
                cx, cy = int(centroid[0]), int(centroid[1])
                try:
                    side_val = _point_side(a, b, (cx, cy))
                except Exception:
                    side_val = None
                prev_side_val = object_prev_side.get(obj_id)
                best_idx = None
                best_dist = float('inf')
                for i, c in enumerate(track_centroids):
                    if i in used:
                        continue
                    d = math.hypot(cx - c[0], cy - c[1])
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
                if best_idx is not None and best_dist < args.max_distance:
                    used.add(best_idx)
                    det_index = track_map[best_idx]
                    x1, y1, x2, y2 = det_boxes[det_index]
                    if box_intersects_line((x1, y1, x2, y2), a, b):
                        centroid_xy = (cx, cy)
                        try:
                            import config as _cfg
                            cfg = _cfg
                        except Exception:
                            cfg = None
                        dedup_cooldown = getattr(cfg, 'DEDUP_COOLDOWN_SECONDS', 6) if cfg else 6
                        require_side = getattr(cfg, 'REQUIRE_SIDE_TRANSITION', True) if cfg else True
                        if tracker.counted.get(obj_id, False):
                            print(f"DEBUG: ID {obj_id} already tracker-counted; skipping")
                            continue
                        now_ts = time.time()
                        stable_obj, created_new, was_reassoc = _associate_to_stable(
                            stable_objects,
                            obj_id,
                            centroid_xy,
                            (x1, y1, x2, y2),
                            now_ts,
                            side_val,
                            max_dist=cfg_track_dist,
                            min_iou=cfg_track_iou,
                            timeout=cfg_track_timeout,
                            max_recent=getattr(config, 'DEDUP_MAX_RECENT', 200) if config else 200,
                            debug=cfg_debug,
                        )
                        if stable_obj is not None:
                            if stable_obj.get('counted'):
                                tracker.counted[obj_id] = True
                                recent_events.append({'ts': now_ts, 'centroid': centroid_xy, 'bbox': (x1, y1, x2, y2), 'obj_id': obj_id, 'sack_number': stable_obj.get('sack_number')})
                                _cleanup_recent(recent_events, dedup_cooldown)
                                continue
                            prev_side = stable_obj.get('last_side')
                            side_ok = False
                            line_tol = getattr(config, 'LINE_CROSSING_TOLERANCE', 8.0) if config else 8.0
                            if require_side:
                                if prev_side is None and prev_side_val is None:
                                    side_ok = True
                                else:
                                    prev = prev_side if prev_side is not None else prev_side_val
                                    if prev is None or side_val is None:
                                        side_ok = False
                                    else:
                                        prev_dist = abs(prev) / line_len
                                        curr_dist = abs(side_val) / line_len
                                        if (prev * side_val) < 0 and (prev_dist >= line_tol or curr_dist >= line_tol):
                                            side_ok = True
                            else:
                                side_ok = True
                            if not side_ok:
                                if cfg_debug:
                                    print(f"DEBUG: ID {obj_id} stable {stable_obj['stable_id']} side check failed -> skip")
                                recent_events.append({'ts': now_ts, 'centroid': centroid_xy, 'bbox': (x1, y1, x2, y2), 'obj_id': obj_id, 'sack_number': stable_obj.get('sack_number')})
                                _cleanup_recent(recent_events, dedup_cooldown)
                                tracker.counted[obj_id] = False
                                continue
                            total_count += 1
                            interval_count += 1
                            tracker.counted[obj_id] = True
                            count_flash[obj_id] = time.time() + 1.2
                            print(f'COUNTED SACK | ID: {obj_id} | TOTAL: {total_count}')
                            try:
                                res = record_sack(obj_id=obj_id, source='rtsp', details=f'stable_id={stable_obj["stable_id"]} box={x1},{y1},{x2},{y2}')
                                if res and isinstance(res, dict):
                                    last_sack_time = res.get('timestamp')
                                    stable_obj['sack_number'] = res.get('sack_number')
                            except Exception:
                                try:
                                    increment(1)
                                    log_event(obj_id=obj_id, source='rtsp', details=f'stable_id={stable_obj["stable_id"]} box={x1},{y1},{x2},{y2}')
                                except Exception:
                                    pass
                            stable_obj['counted'] = True
                            recent_events.append({'ts': now_ts, 'centroid': centroid_xy, 'bbox': (x1, y1, x2, y2), 'obj_id': obj_id, 'sack_number': stable_obj.get('sack_number')})
                            _cleanup_recent(recent_events, dedup_cooldown)
                            continue
                        stable_id = _stable_counter['v']
                        _stable_counter['v'] += 1
                        new_stable = {
                            'stable_id': stable_id,
                            'current_track_id': obj_id,
                            'previous_track_ids': set([obj_id]),
                            'bbox': (x1, y1, x2, y2),
                            'center': centroid_xy,
                            'last_seen_ts': now_ts,
                            'history': [{'center': centroid_xy, 'ts': now_ts, 'bbox': (x1, y1, x2, y2)}],
                            'velocity': (0.0, 0.0),
                            'last_side': side_val,
                            'counted': False,
                            'sack_number': None,
                            'class': target_cls,
                        }
                        stable_objects.append(new_stable)
                        max_stable = getattr(config, 'DEDUP_MAX_RECENT', 200) if config else 200
                        if len(stable_objects) > max_stable:
                            stable_objects.pop(0)
                        total_count += 1
                        interval_count += 1
                        tracker.counted[obj_id] = True
                        count_flash[obj_id] = time.time() + 1.2
                        print(f'COUNTED SACK | ID: {obj_id} | TOTAL: {total_count}')
                        try:
                            res = record_sack(obj_id=obj_id, source='rtsp', details=f'stable_id={stable_id} box={x1},{y1},{x2},{y2}')
                            if res and isinstance(res, dict):
                                last_sack_time = res.get('timestamp')
                                new_stable['sack_number'] = res.get('sack_number')
                        except Exception:
                            try:
                                increment(1)
                                log_event(obj_id=obj_id, source='rtsp', details=f'stable_id={stable_id} box={x1},{y1},{x2},{y2}')
                            except Exception:
                                pass
                        new_stable['counted'] = True
                        recent_events.append({'ts': now_ts, 'centroid': centroid_xy, 'bbox': (x1, y1, x2, y2), 'obj_id': obj_id, 'sack_number': new_stable.get('sack_number')})
                        _cleanup_recent(recent_events, dedup_cooldown)
                        continue
                try:
                    object_prev_side[obj_id] = side_val
                except Exception:
                    object_prev_side[obj_id] = None

            if frames_processed == 1:
                print('FIRST COUNTING FRAME PROCESSED')
                print('COUNTING COMPLETE')

            now = time.time()
            if now - last_update >= args.update_interval:
                text = f'Sacks unloaded (last {args.update_interval}s): {interval_count} — Total: {total_count}'
                try:
                    annotated = annotate_frame(frame, det_boxes, objects, objects, a, b, total_count, interval_count, labels=det_labels, statuses=det_statuses, cfg_debug=cfg_debug)
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

            try:
                now_ts = time.time()
                active_highlights = {obj_id for obj_id, expiry in list(count_flash.items()) if expiry > now_ts}
                annotated_frame = annotate_frame(frame, det_boxes, objects, objects, a, b, total_count, interval_count, labels=det_labels, statuses=det_statuses, cfg_debug=cfg_debug, highlight_ids=active_highlights)
                last_annotated = annotated_frame
                try:
                    fps = 0.0
                    if 'last_frame_time' in globals():
                        elapsed = time.time() - globals()['last_frame_time']
                        fps = 1.0 / max(elapsed, 1e-3)
                except Exception:
                    fps = 0.0
                globals()['last_frame_time'] = time.time()
                cv2.putText(annotated_frame, f'SACK COUNT: {total_count}', (20, annotated_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'DETECTIONS: {len(det_boxes)}', (20, annotated_frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f'TRACKED: {len(objects)}', (20, annotated_frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (20, annotated_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if frames_processed == 1:
                    print('FIRST PROCESSED FRAME READY')
                print('ABOUT TO DISPLAY FRAME')
                cv2.imshow('Sack Counter - Live', annotated_frame)
                live_window_ready = True
                if frames_processed == 1:
                    print('LIVE WINDOW DISPLAYING')
                try:
                    if recorder is not None:
                        recorder.write_frame(annotated_frame)
                except Exception:
                    pass
                if frames_processed % 100 == 0:
                    print(f'Frames: {frames_processed} | Detections: {len(det_boxes)} | Tracked: {len(objects)} | Sack Count: {total_count}')
            except Exception as e:
                print('ANNOTATION/IMSHOW EXCEPTION:')
                print(repr(e))
                continue
            print('WAITKEY COMPLETE')
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print('Quit key pressed, stopping live processing...')
                stop_requested = True
    except KeyboardInterrupt:
        print('Keyboard interrupt received — shutting down...')
    except Exception as e:
        print('Unexpected error in main loop:', e)
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    finally:
        # Finalization / graceful shutdown
        print('Processing stopped.')
        print(f'FINAL SACK COUNT: {total_count}')
        print('Finalizing recording...')
        recording_path = None
        saved_videos = []
        try:
            if recorder is not None:
                try:
                    recording_path = getattr(recorder, 'segment_path', None)
                    rec_dir = getattr(recorder, 'directory', None)
                    if rec_dir and os.path.isdir(rec_dir):
                        files = [os.path.join(rec_dir, f) for f in os.listdir(rec_dir) if f.lower().endswith('.mp4')]
                        files = sorted(files, key=lambda p: os.path.getmtime(p))
                        if files:
                            saved_videos = files
                            if recording_path is None:
                                recording_path = files[-1]
                    try:
                        recorder.stop()
                    except Exception:
                        pass
                    if rec_dir and os.path.isdir(rec_dir):
                        files2 = [os.path.join(rec_dir, f) for f in os.listdir(rec_dir) if f.lower().endswith('.mp4')]
                        files2 = sorted(files2, key=lambda p: os.path.getmtime(p))
                        if files2:
                            saved_videos = files2
                            if recording_path is None or not os.path.exists(recording_path):
                                recording_path = files2[-1]
                except Exception as e:
                    print('Error while finalizing recorder:', e)
        except Exception:
            pass

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if recorder is not None:
                recorder.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if recording_path and os.path.exists(recording_path):
            print(f'RECORDING SAVED: {recording_path}')
        elif saved_videos:
            print(f'RECORDING SAVED: {saved_videos[-1]}')

        print('Shutdown complete.')

        # send Telegram final report
        try:
            print('Sending Telegram report...')
            text = f'Final sack count: {total_count}'
            photo = None
            try:
                if last_annotated is not None:
                    fd, tmpf = tempfile.mkstemp(suffix='.jpg')
                    os.close(fd)
                    cv2.imwrite(tmpf, last_annotated)
                    photo = tmpf
            except Exception:
                photo = None

            tg_ok = False
            try:
                tg_ok = send_telegram(args.telegram_token, args.telegram_chat_id, text, photo_path=photo)
            except Exception as e:
                print('Telegram send error during shutdown:', e)
                tg_ok = False
            if photo and os.path.exists(photo):
                try:
                    os.remove(photo)
                except Exception:
                    pass
            if tg_ok:
                print('Telegram report sent successfully.')
            else:
                print('Telegram report failed. Video retained.')
        except Exception as e:
            print('Failed to send Telegram report:', e)

        print('Shutdown complete.')


if __name__ == '__main__':
    main()
