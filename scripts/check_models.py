"""List the Gemini models this API key can see, then ping each Flash-class one to confirm it
actually answers on this key's tier (listing a model doesn't guarantee free-tier quota).

    uv run python scripts/check_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402

from app import config  # noqa: E402

client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(
    timeout=20_000, retry_options=types.HttpRetryOptions(attempts=1)))
names = sorted(m.name.removeprefix("models/") for m in client.models.list()
               if "generateContent" in (m.supported_actions or []))
print(f"{len(names)} models support generateContent:")
for n in names:
    print("  ", n)

flash = [n for n in names if "flash" in n and not any(x in n for x in ("image", "tts", "audio", "live", "thinking"))]
print("\nPinging Flash-class models:")
for n in flash:
    try:
        r = client.models.generate_content(model=n, contents="Reply with the single word: ok")
        print(f"  OK   {n}: {r.text.strip()[:20]!r}")
    except errors.APIError as e:
        print(f"  FAIL {n}: {e.code} {str(e.message)[:80]}")
    except Exception as e:  # timeouts etc.
        print(f"  FAIL {n}: {type(e).__name__}")
