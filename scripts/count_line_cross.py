#!/usr/bin/env python3
"""
Count objects of a given class crossing a line in a video.

Dependencies:
  pip install ultralytics opencv-python numpy

Usage example:
  python scripts/count_line_cross.py \
    --source runs/detect/predict-2/IMG_1047.mp4 \
    --weights yolo11n.pt \
    --class_name Sack \
    --line 0.5,0.0,0.5,1.0 \
    --output runs/detect/predict-2/IMG_1047_counted.mp4

The `--line` argument expects normalized coordinates x1,y1,x2,y2 (0..1) relative to frame size.

"""
import argparse
import math
import sys
from collections import OrderedDict

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class CentroidTracker:
    def __init__(self, max_disappeared=60, max_distance=80, min_iou=0.3):
        self.next_object_id = 0
        self.objects = OrderedDict()  # object_id -> centroid
        self.last_box = {}  # object_id -> last box (x1,y1,x2,y2)
        self.disappeared = OrderedDict()  # object_id -> frames disappeared
        self.counted = {}  # object_id -> bool
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_iou = min_iou

    def register(self, centroid, box=None):
        self.objects[self.next_object_id] = centroid
        self.last_box[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.counted[self.next_object_id] = False
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        if object_id in self.last_box:
            del self.last_box[object_id]
        del self.disappeared[object_id]
        del self.counted[object_id]

    def _iou(self, boxA, boxB):
        if boxA is None or boxB is None:
            return 0.0
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
        boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
        if boxAArea + boxBArea - interArea <= 0:
            return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)


    def update(self, input_centroids, input_boxes=None):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                b = input_boxes[i] if input_boxes is not None and i < len(input_boxes) else None
                self.register(c, box=b)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # prefer IoU based matching using last_box; fallback to centroid distance
        IOU = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, obj_id in enumerate(object_ids):
            boxA = self.last_box.get(obj_id, None)
            for j, boxB in enumerate(input_boxes if input_boxes is not None else [None]*len(input_centroids)):
                IOU[i, j] = self._iou(boxA, boxB) if boxA is not None and boxB is not None else 0.0

        used_rows = set()
        used_cols = set()

        # match by IoU first
        for _ in range(min(IOU.shape[0], IOU.shape[1])):
            i, j = divmod(IOU.argmax(), IOU.shape[1])
            if IOU[i, j] <= 0:
                break
            object_id = object_ids[i]
            self.objects[object_id] = input_centroids[j]
            self.last_box[object_id] = input_boxes[j]
            self.disappeared[object_id] = 0
            used_rows.add(i)
            used_cols.add(j)
            IOU[i, :] = -1
            IOU[:, j] = -1

        # remaining unmatched -> distance matching
        remaining_obj_idxs = [i for i in range(len(object_centroids)) if i not in used_rows]
        remaining_det_idxs = [j for j in range(len(input_centroids)) if j not in used_cols]

        if remaining_obj_idxs and remaining_det_idxs:
            D = np.zeros((len(remaining_obj_idxs), len(remaining_det_idxs)), dtype=float)
            for ii, i in enumerate(remaining_obj_idxs):
                oc = object_centroids[i]
                for jj, j in enumerate(remaining_det_idxs):
                    ic = input_centroids[j]
                    D[ii, jj] = np.linalg.norm(np.array(oc) - np.array(ic))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            for (row, col) in zip(rows, cols):
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[remaining_obj_idxs[row]]
                det_idx = remaining_det_idxs[cols[row]]
                self.objects[object_id] = input_centroids[det_idx]
                self.last_box[object_id] = input_boxes[det_idx] if input_boxes is not None else None
                self.disappeared[object_id] = 0
                used_rows.add(remaining_obj_idxs[row])
                used_cols.add(det_idx)


        unused_rows = set(range(0, len(object_centroids))) - used_rows
        unused_cols = set(range(0, len(input_centroids))) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            b = input_boxes[col] if input_boxes is not None and col < len(input_boxes) else None
            self.register(input_centroids[col], box=b)

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
    parser.add_argument('--weights', default='yolo11n.pt', help='model weights')
    parser.add_argument('--output', default=None, help='output video path')
    parser.add_argument('--class_name', default='Sack', help='class name to count')
    parser.add_argument('--class_id', type=int, default=None, help='class id to count (overrides class_name)')
    parser.add_argument('--line', default='0.5,0.0,0.5,1.0', help='normalized line x1,y1,x2,y2')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--max-distance', type=int, default=60, help='max matching distance in pixels')
    args = parser.parse_args()

    if YOLO is None:
        print('Please install ultralytics: pip install ultralytics')
        sys.exit(1)

    model = YOLO(args.weights)

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

    lx1, ly1, lx2, ly2 = parse_line(args.line)
    a = (int(lx1 * width), int(ly1 * height))
    b = (int(lx2 * width), int(ly2 * height))

    tracker = CentroidTracker(max_distance=args.max_distance)
    object_prev_side = {}
    total_count = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # run detection
        try:
            results = model.predict(source=frame, imgsz=640, conf=args.conf, device='')
        except Exception:
            results = model(frame)

        # ultralytics returns a list-like of results; take first
        r = results[0]

        boxes = []
        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            # fallback if boxes attribute differs
            try:
                xyxy = r.boxes.xyxy.numpy()
                confs = r.boxes.conf.numpy()
                clss = r.boxes.cls.numpy().astype(int)
            except Exception:
                xyxy = np.array([])
                confs = np.array([])
                clss = np.array([])

        # choose which class id to count
        if args.class_id is not None:
            target_cls = args.class_id
        else:
            # try to map name -> id using model.names if available
            try:
                names = model.names
                target_cls = None
                for k, v in names.items():
                    if v.lower() == args.class_name.lower():
                        target_cls = int(k)
                        break
                if target_cls is None:
                    # fallback: try common index 1 for 'Sack'
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

        objects = tracker.update(det_centroids, det_boxes)

        # map current centroids back to boxes for drawing and detect intersection
        used = set()
        for obj_id, centroid in objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])

            # find matching detection box (closest)
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

                # count if any part of the box touches the line
                if box_intersects_line((x1, y1, x2, y2), a, b):
                    if not tracker.counted.get(obj_id, False):
                        total_count += 1
                        tracker.counted[obj_id] = True

            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # draw line and overlay
        cv2.line(frame, a, b, (0, 0, 255), 2)
        cv2.putText(frame, f'Count: {total_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print('Done. Output saved to', args.output)


if __name__ == '__main__':
    main()
