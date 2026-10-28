import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

OFFICE_NAME = "Bright Smile Dental"
OFFICE_PHONE = "(555) 010-2200"
OFFICE_ADDRESS = "1450 Ocean Park Blvd, Suite 200, Santa Monica, CA"  # placeholder address
OFFICE_TZ = ZoneInfo(os.getenv("OFFICE_TZ") or "America/Los_Angeles")
OPEN_HOUR, CLOSE_HOUR = 8, 18
OPEN_WEEKDAYS = range(6)  # Mon-Sat
SLOT_LEN = timedelta(minutes=30)
APPT_TYPES = ("cleaning", "checkup", "emergency")

DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./dental.db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
STAFF_TOKEN = os.getenv("STAFF_TOKEN", "")
LLM_MAX_RPM = float(os.getenv("LLM_MAX_RPM") or 0)  # 0 = no client-side throttle


def office_now() -> datetime:
    """Naive wall-clock time at the office. All slot times are stored the same way."""
    # ponytail: naive local times work for one office; store UTC + per-office tz for multi-location
    return datetime.now(OFFICE_TZ).replace(tzinfo=None, second=0, microsecond=0)
