#!/usr/bin/env python3
"""
Simple launcher for the counting scripts using centralized `config.py`.

Usage:
  python app.py rtsp    # run RTSP counting service using config
  python app.py offline --source path/to/video.mp4  # run offline counting

This file does not modify detection or line-crossing math; it only prepares
command-line arguments from `config.py` (or environment) and invokes the
existing scripts.
"""
import sys
import runpy
import argparse
import os
import subprocess
import time
import logging
import signal
import sys

try:
    import config
except Exception:
    config = None


def run_rtsp():
    # Build command to run the RTSP server as a separate process
    cmd = [sys.executable, os.path.join('scripts', 'rtsp_count_server.py')]
    if config and config.RTSP_URL:
        cmd += ['--rtsp', config.RTSP_URL]
    if config and config.WEIGHTS:
        cmd += ['--weights', config.WEIGHTS]
    if config and config.LINE_RTSP:
        cmd += ['--line', config.LINE_RTSP]
    if config and config.CONF_RTSP is not None:
        cmd += ['--conf', str(config.CONF_RTSP)]
    if config and config.CLASS_NAME:
        cmd += ['--class_name', config.CLASS_NAME]
    if config and config.CLASS_ID is not None:
        cmd += ['--class_id', str(config.CLASS_ID)]
    if config and config.TELEGRAM_TOKEN:
        cmd += ['--telegram-token', config.TELEGRAM_TOKEN]
    if config and config.TELEGRAM_CHAT_ID:
        cmd += ['--telegram-chat-id', config.TELEGRAM_CHAT_ID]
    if config and config.UPDATE_INTERVAL:
        cmd += ['--update-interval', str(config.UPDATE_INTERVAL)]
    if config and config.MAX_DISTANCE:
        cmd += ['--max-distance', str(config.MAX_DISTANCE)]
    if config and config.SAVE_SNAPSHOTS:
        cmd += ['--save-snapshots']

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    def _terminate_process(p):
        try:
            if p and p.poll() is None:
                logging.info('Terminating child process...')
                p.terminate()
                try:
                    p.wait(timeout=5)
                except Exception:
                    logging.info('Killing child process...')
                    p.kill()
        except Exception:
            pass

    logging.info('Starting RTSP runner loop')
    while True:
        proc = None
        try:
            logging.info('Launching: %s', ' '.join(cmd))
            proc = subprocess.Popen(cmd)
            # Wait for child to exit; if it exits (e.g., due to camera disconnect or error), restart
            ret = proc.wait()
            logging.warning('RTSP process exited with code %s; will restart in 5s', ret)
        except KeyboardInterrupt:
            logging.info('Received KeyboardInterrupt, shutting down')
            _terminate_process(proc)
            raise
        except Exception:
            logging.exception('Error while running RTSP process; will retry in 5s')
            _terminate_process(proc)
        finally:
            # ensure resources are released; give time for OS to clean up
            try:
                if proc and proc.poll() is None:
                    _terminate_process(proc)
            except Exception:
                pass
        time.sleep(5)


def run_offline(source=None, output=None):
    argv = ["count_line_cross.py"]
    if source:
        argv += ["--source", source]
    if config and config.WEIGHTS:
        argv += ["--weights", config.WEIGHTS]
    if config and config.LINE_OFFLINE:
        argv += ["--line", config.LINE_OFFLINE]
    if config and config.CONF_OFFLINE is not None:
        argv += ["--conf", str(config.CONF_OFFLINE)]
    if config and config.CLASS_NAME:
        argv += ["--class_name", config.CLASS_NAME]
    if config and config.CLASS_ID is not None:
        argv += ["--class_id", str(config.CLASS_ID)]
    if config and config.MAX_DISTANCE:
        argv += ["--max-distance", str(config.MAX_DISTANCE)]
    if output:
        argv += ["--output", output]
    # Ensure we proceed past the line-check when launching from app.py
    argv += ["--proceed"]

    sys.argv = argv
    runpy.run_path(os.path.join('scripts', 'count_line_cross.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode')
    sub.add_parser('rtsp')
    off = sub.add_parser('offline')
    off.add_argument('--source', help='video file to process')
    off.add_argument('--output', help='output video file')

    args = parser.parse_args()
    if args.mode == 'rtsp':
        run_rtsp()
    elif args.mode == 'offline':
        if not args.source:
            print('offline mode requires --source')
            sys.exit(1)
        run_offline(source=args.source, output=args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
