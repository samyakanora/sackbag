import os
import time
import shutil
import threading
from datetime import datetime, timedelta
import cv2


class Recorder:
    def __init__(self, directory='recordings', segment_minutes=10, retention_days=1):
        self.directory = directory
        self.segment_minutes = int(segment_minutes)
        self.retention_days = int(retention_days)
        os.makedirs(self.directory, exist_ok=True)
        self.writer = None
        self.segment_start = None
        self.segment_path = None
        self.lock = threading.Lock()
        self.last_cleanup = datetime.utcnow()

    def _cleanup_old(self):
        # remove recordings older than retention_days
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        for fn in os.listdir(self.directory):
            path = os.path.join(self.directory, fn)
            try:
                mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
                if mtime < cutoff:
                    os.remove(path)
            except Exception:
                pass

    def _disk_ok(self, min_free_mb=200):
        try:
            total, used, free = shutil.disk_usage(self.directory)
            free_mb = free // (1024 * 1024)
            return free_mb >= min_free_mb
        except Exception:
            return True

    def _start_segment(self, frame_size, fps=20.0):
        # ensure cleanup before starting
        try:
            now = datetime.utcnow()
            self._cleanup_old()
            self.segment_start = now
            fname = now.strftime('rec_%Y%m%d_%H%M%S.mp4')
            self.segment_path = os.path.join(self.directory, fname)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.segment_path, fourcc, fps, frame_size)
        except Exception:
            self.writer = None

    def write_frame(self, frame):
        if frame is None:
            return
        with self.lock:
            # periodic cleanup
            if (datetime.utcnow() - self.last_cleanup).seconds > 60:
                self._cleanup_old()
                self.last_cleanup = datetime.utcnow()

            if not self._disk_ok():
                # disk low: stop recording
                self._stop_writer()
                return

            h, w = frame.shape[:2]
            frame_size = (w, h)
            # start writer if needed
            if self.writer is None:
                self._start_segment(frame_size)
            # rotate segment if needed
            if self.segment_start:
                elapsed = (datetime.utcnow() - self.segment_start).total_seconds()
                if elapsed >= (self.segment_minutes * 60):
                    self._stop_writer()
                    self._start_segment(frame_size)
            # write frame
            try:
                if self.writer:
                    self.writer.write(frame)
            except Exception:
                pass

    def _stop_writer(self):
        writer = self.writer
        self.writer = None
        self.segment_start = None
        self.segment_path = None
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

    def stop(self):
        with self.lock:
            self._stop_writer()
