import os

# Core model and source settings
WEIGHTS = os.getenv('WEIGHTS', 'runs/detect/train/weights/best.pt')
# RTSP URL should be provided via environment variable or external config.
# Do NOT hardcode passwords in this file. Default is empty; set RTSP_URL via env.
RTSP_URL = os.getenv('RTSP_URL', '')

# Line definitions.
# LINE_OFFLINE remains normalized for offline reference frames.
# LINE_RTSP is in actual RTSP processing-pixel coordinates for the 2560x1440 stream.
LINE_OFFLINE = os.getenv('LINE_OFFLINE', '12,227,467,0')
LINE_RTSP = os.getenv('LINE_RTSP', '870,1400,2340,0')

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

# Recording / storage configuration
RECORDING_ENABLED = os.getenv('RECORDING_ENABLED', 'True') not in ('0', 'False', '', 'false')
RECORDING_DIRECTORY = os.getenv('RECORDING_DIRECTORY', 'recordings')
RECORDING_SEGMENT_MINUTES = int(os.getenv('RECORDING_SEGMENT_MINUTES', '10'))
RECORDING_RETENTION_DAYS = int(os.getenv('RECORDING_RETENTION_DAYS', '1'))

# Operational monitoring
SACK_MILESTONE = int(os.getenv('SACK_MILESTONE', '35'))
IDLE_TIMEOUT = int(os.getenv('IDLE_TIMEOUT', '60'))  # seconds to consider idle

# Database location
SACK_DB_PATH = os.getenv('SACK_DB_PATH', os.path.join(os.path.dirname(__file__), 'sack_counts.db'))

# Deduplication and crossing behavior
DEDUP_COOLDOWN_SECONDS = int(os.getenv('DEDUP_COOLDOWN_SECONDS', '6'))
DEDUP_DISTANCE_PIXELS = int(os.getenv('DEDUP_DISTANCE_PIXELS', '80'))
DEDUP_MAX_RECENT = int(os.getenv('DEDUP_MAX_RECENT', '200'))
REQUIRE_SIDE_TRANSITION = os.getenv('REQUIRE_SIDE_TRANSITION', 'True') not in ('0', 'False', 'false', '')

# Stable tracking / reassociation tuning
TRACK_REASSOCIATION_DISTANCE = int(os.getenv('TRACK_REASSOCIATION_DISTANCE', '80'))
TRACK_REASSOCIATION_IOU = float(os.getenv('TRACK_REASSOCIATION_IOU', '0.3'))
TRACK_REASSOCIATION_TIMEOUT = float(os.getenv('TRACK_REASSOCIATION_TIMEOUT', '1.5'))

# Minimum detection size to consider (pixels)
MIN_PRODUCT_WIDTH = int(os.getenv('MIN_PRODUCT_WIDTH', '20'))
MIN_PRODUCT_HEIGHT = int(os.getenv('MIN_PRODUCT_HEIGHT', '20'))
MIN_PRODUCT_AREA = int(os.getenv('MIN_PRODUCT_AREA', '400'))

# Aspect ratio limits for detected product (width/height)
MIN_PRODUCT_ASPECT_MIN = float(os.getenv('MIN_PRODUCT_ASPECT_MIN', '0.4'))
MIN_PRODUCT_ASPECT_MAX = float(os.getenv('MIN_PRODUCT_ASPECT_MAX', '2.5'))

# Person overlap filtering: IoU threshold above which a product detection
# that is largely contained in a person box may be considered ambiguous.
PERSON_OVERLAP_THRESHOLD = float(os.getenv('PERSON_OVERLAP_THRESHOLD', '0.5'))

# Line crossing tolerance in pixels (distance from line). Helps avoid jitter.
LINE_CROSSING_TOLERANCE = float(os.getenv('LINE_CROSSING_TOLERANCE', '8.0'))

# Tracking tuning
TRACK_REASSOCIATION_TIMEOUT = float(os.getenv('TRACK_REASSOCIATION_TIMEOUT', '2.0'))
# how long an object can be disappeared before deregistering (frames*2 in CentroidTracker)
TRACK_MAX_DISAPPEARED = int(os.getenv('TRACK_MAX_DISAPPEARED', '30'))

# Debugging
DEBUG_TRACKING = os.getenv('DEBUG_TRACKING', 'False') not in ('0', 'False', 'false', '')
