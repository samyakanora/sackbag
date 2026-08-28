#!/usr/bin/env python3
"""
Count `Sack` objects (class_id=1) crossing a scaled counting line using
Ultralytics ByteTrack for persistent IDs.

Dependencies:
    pip install ultralytics opencv-python numpy

Usage example (app launcher):
    python app.py offline --source 20260827170855041.MP4

This script scales a reference counting line defined in a 582x328
coordinate system to the actual input video resolution so that the
physical line position is preserved across resolutions.
"""
import argparse
import math
import sys
import time
from collections import OrderedDict
import os
import cv2
import numpy as np
from datetime import datetime

try:
    # storage not used for offline visual test, but keep import defensive
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

import csv


# ByteTrack will be used via Ultralytics `model.track(..., tracker='bytetrack')`.
# We keep our own minimal bookkeeping for counting and "just counted" display.


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
    # rectangle corners
    rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    # check intersection with any edge
    for i in range(4):
        if _segments_intersect(a, b, rect[i], rect[(i + 1) % 4]):
            return True
    # also check if line endpoints are inside the box
    if x1 <= a[0] <= x2 and y1 <= a[1] <= y2:
        return True
    if x1 <= b[0] <= x2 and y1 <= b[1] <= y2:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='input video path')
    # load defaults from config if present
    try:
        import config
    except Exception:
        config = None

    parser.add_argument('--weights', default=(config.WEIGHTS if config else 'yolo11n.pt'), help='model weights')
    parser.add_argument('--output', default=None, help='output video path')
    parser.add_argument('--class_name', default=(config.CLASS_NAME if config else 'Sack'), help='class name to count')
    parser.add_argument('--class_id', type=int, default=(config.CLASS_ID if config else None), help='class id to count (overrides class_name)')
    parser.add_argument('--line', default=(config.LINE_OFFLINE if config else '0.5,0.0,0.5,1.0'), help='normalized line x1,y1,x2,y2')
    parser.add_argument('--conf', type=float, default=(config.CONF_OFFLINE if config else 0.25), help='confidence threshold')
    parser.add_argument('--max-distance', type=int, default=(config.MAX_DISTANCE if config else 60), help='max matching distance in pixels')
    parser.add_argument('--proceed', action='store_true', help='Proceed past first-frame line check and run full video processing')
    args = parser.parse_args()

    if YOLO is None:
        print('Please install ultralytics: pip install ultralytics')
        sys.exit(1)

    model = YOLO(args.weights)
    # print full class mapping for verification
    try:
        print('Model class mapping:', model.names)
    except Exception:
        pass

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print('Failed to open', args.source)
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if args.output is None:
        args.output = args.source.rsplit('.', 1)[0] + '_counted.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # The user provided a reference resolution and reference coordinates.
    # Accept line in either reference-pixel coords (e.g. '0,185,530,0') or
    # normalized coords 'x1,y1,x2,y2' (0..1). Default reference coords below.
    # Use reference resolution scaling so the physical line is preserved.
    # Default reference coords per user: (0,185) -> (530,0) with ref 581x328.
    ref_w = getattr(config, 'REFERENCE_WIDTH', 582) if config else 582
    ref_h = getattr(config, 'REFERENCE_HEIGHT', 328) if config else 328

    # If the provided --line is a string of integers (reference pixels), use them.
    try:
        lx1, ly1, lx2, ly2 = parse_line(args.line)
    except Exception:
        # fallback normalized parsing
        lx1, ly1, lx2, ly2 = 0.0, 0.0, 0.0, 0.0

    # determine whether values are normalized (0..1) or reference-pixel coords
    if 0.0 <= lx1 <= 1.0 and 0.0 <= ly1 <= 1.0 and 0.0 <= lx2 <= 1.0 and 0.0 <= ly2 <= 1.0:
        # normalized coordinates relative to frame size
        a = (int(lx1 * width), int(ly1 * height))
        b = (int(lx2 * width), int(ly2 * height))
    else:
        # treat as reference-pixel coords and scale to actual frame
        a = (int(lx1 * width / ref_w), int(ly1 * height / ref_h))
        b = (int(lx2 * width / ref_w), int(ly2 * height / ref_h))

    # Save and show first frame with scaled counting line for verification.
    cap2 = cv2.VideoCapture(args.source)
    ok, first = cap2.read()
    if not ok:
        print('Failed to read first frame for line check; aborting')
        cap2.release()
        return
    # draw scaled line on first frame
    frame_check = first.copy()
    cv2.line(frame_check, a, b, (0, 0, 255), 3)
    # overlay info
    cv2.putText(frame_check, 'SCALED LINE CHECK', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    os.makedirs('runs/detect/verify', exist_ok=True)
    check_path = os.path.join('runs', 'detect', 'verify', 'line_check_first_frame.jpg')
    cv2.imwrite(check_path, frame_check)
    print('Saved first-frame line check image to', check_path)
    print('Inspect the image and re-run with --proceed to process the full video if line placement is correct.')
    cap2.release()
    if not args.proceed:
        return

    # Use ByteTrack via Ultralytics tracker. Start counters fresh for offline test.
    total_count = 0
    counted_track_ids = set()
    unique_ids = set()
    total_sack_detections = 0
    just_counted_timers = {}  # track_id -> frames remaining to show 'COUNTED'
    SHOW_COUNTED_FRAMES = 30

    # prepare event CSV for offline debug (overwrite existing)
    events_path = os.path.join('runs', 'detect', 'verify', 'count_events.csv')
    os.makedirs(os.path.dirname(events_path), exist_ok=True)
    with open(events_path, 'w', newline='') as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(['frame_idx', 'track_id', 'x1', 'y1', 'x2', 'y2', 'conf'])

    # choose which class id to count (force to 1 if not provided)
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

    # iterate using the model.track stream so ByteTrack assigns persistent IDs
    # Ultralytics expects a tracker YAML filename ('.yaml' or '.yml').
    stream = model.track(source=args.source, tracker='bytetrack.yaml', stream=True, conf=args.conf, imgsz=640)
    frame_idx = 0
    for r in stream:
        # r is a Results object for one frame
        frame_idx += 1
        try:
            frame = r.orig_img if hasattr(r, 'orig_img') else r.orig_img
        except Exception:
            # fallback to OpenCV read if not available
            ret, frame = cap.read()
            if not ret:
                break

        # extract boxes, classes, confidences, and track ids
        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int)
        except Exception:
            try:
                xyxy = r.boxes.xyxy.numpy()
                confs = r.boxes.conf.numpy()
                clss = r.boxes.cls.numpy().astype(int)
                ids = r.boxes.id.numpy().astype(int)
            except Exception:
                xyxy = np.array([])
                confs = np.array([])
                clss = np.array([])
                ids = np.array([])

        # process and draw all detections; count only Sack class
        for i, box in enumerate(xyxy):
            if i >= len(confs):
                continue
            if confs[i] < args.conf:
                continue
            x1, y1, x2, y2 = box.astype(int)
            cls_id = int(clss[i]) if i < len(clss) else None
            tid = int(ids[i]) if i < len(ids) else None

            # class name
            try:
                cls_name = model.names.get(cls_id, str(cls_id)) if model and hasattr(model, 'names') else str(cls_id)
            except Exception:
                cls_name = str(cls_id)

            # draw bbox and labels for all classes
            color = (0, 255, 0) if cls_name.lower() == 'sack' or cls_id == target_cls else (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{cls_name} {confs[i]:.2f}'
            cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            id_text = f'ID: {tid}' if tid is not None else 'ID: -'
            cv2.putText(frame, id_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # bookkeeping
            if cls_id == target_cls and tid is not None:
                total_sack_detections += 1
                unique_ids.add(tid)
                # check rectangle-line intersection
                if box_intersects_line((x1, y1, x2, y2), a, b):
                    if tid not in counted_track_ids:
                        total_count += 1
                        counted_track_ids.add(tid)
                        just_counted_timers[tid] = SHOW_COUNTED_FRAMES
                        # log event to CSV for later frame extraction
                        try:
                            with open(events_path, 'a', newline='') as fh:
                                w = csv.writer(fh)
                                w.writerow([frame_idx, tid, x1, y1, x2, y2, f'{confs[i]:.4f}'])
                        except Exception:
                            pass

        # draw counting line
        cv2.line(frame, a, b, (0, 0, 255), 3)

        # draw big sack count overlay
        overlay_text = f'SACK COUNT: {total_count}'
        # large text near top-right
        tw, th = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)[0]
        cv2.putText(frame, overlay_text, (max(10, width - tw - 20), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)

        # draw 'COUNTED' markers for recently counted ids
        for tid in list(just_counted_timers.keys()):
            if just_counted_timers[tid] <= 0:
                del just_counted_timers[tid]
                continue
            # find box for this id to annotate; search in current frame boxes
            for i, box in enumerate(xyxy):
                if i < len(ids) and int(ids[i]) == tid:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.putText(frame, 'COUNTED', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
            just_counted_timers[tid] -= 1

        # write frame
        writer.write(frame)

    # release resources
    try:
        cap.release()
    except Exception:
        pass
    writer.release()

    # print final report
    print('\nFinal Report:')
    print('Model:', args.weights)
    print('Class ID:', target_cls)
    print('Tracker: ByteTrack (via ultralytics)')
    print('Confidence threshold:', args.conf)
    print('Reference resolution: {}x{}'.format(ref_w, ref_h))
    print('Actual video resolution: {}x{}'.format(width, height))
    print('Scaled LINE_START:', a)
    print('Scaled LINE_END:  ', b)
    print('Number of Sack detections (frames filtered by conf & cls):', total_sack_detections)
    print('Number of unique ByteTrack IDs seen:', len(unique_ids))
    print('Number of Sack IDs touching/crossing line:', len(counted_track_ids))
    print('Final SACK COUNT:', total_count)
    print('Output video path:', args.output)
    print('\nDone. Output saved to', args.output)


if __name__ == '__main__':
    main()
