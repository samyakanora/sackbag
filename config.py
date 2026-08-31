import os

# Core model and source settings
WEIGHTS = os.getenv('WEIGHTS', 'runs/detect/train/weights/best.pt')
# Default RTSP URL (percent-encode any '@' in the password as %40).
# Provides a sensible default when RTSP_URL env var is not set.
RTSP_URL = os.getenv('RTSP_URL', 'rtsp://admin:Armill%4012@192.168.31.113:554/Streaming/Channels/101')

# Line definitions (normalized x1,y1,x2,y2 strings)
# Use reference-pixel coords for the offline counting line by default
# Reference resolution: 582 x 328
LINE_OFFLINE = os.getenv('LINE_OFFLINE', '12,227,467,0')
LINE_RTSP = os.getenv('LINE_RTSP', '12,227,467,0')

# Reference resolution for scaling the reference line to actual frames
REFERENCE_WIDTH = int(os.getenv('REFERENCE_WIDTH', '582'))
REFERENCE_HEIGHT = int(os.getenv('REFERENCE_HEIGHT', '328'))

# Detection thresholds
CONF_OFFLINE = float(os.getenv('CONF_OFFLINE', os.getenv('CONF', '0.40')))
CONF_RTSP = float(os.getenv('CONF_RTSP', os.getenv('CONF', '0.35')))

# Class settings
CLASS_NAME = os.getenv('CLASS_NAME', 'Sack')
CLASS_ID = os.getenv('CLASS_ID', None)
if CLASS_ID is not None and CLASS_ID != '':
    try:
        CLASS_ID = int(CLASS_ID)
    except Exception:
        CLASS_ID = None

# Tracker and timing
MAX_DISTANCE = int(os.getenv('MAX_DISTANCE', '60'))
UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', '30'))

# Telegram
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Misc
SAVE_SNAPSHOTS = os.getenv('SAVE_SNAPSHOTS', '0') not in ('0', 'False', '', 'false')
