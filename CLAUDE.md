# Bright Smile Dental chatbot

Take-home assessment (brief: `assessment.pdf`, gitignored). Front-desk chatbot for one fictional
dental office: open 8am-6pm Mon-Sat, America/Los_Angeles, (555) 010-2200.

## Core principle
The LLM decides *which* tool to call. Code decides everything that must be correct: dates,
availability, bookings, identity, emergency routing. The model never computes dates, invents slots,
or claims anything a tool didn't confirm.

## Stack (decided, don't relitigate)
- Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2, pytest, uv.
- LLM chain (`LLM_CHAIN`, default `claude,gemini`): Claude Haiku → Gemini Flash-Lite → canned reply
  with the office phone. Providers without a key are skipped; model names come only from env vars.
- DB: SQLite locally, in tests and in evals; Supabase Postgres (transaction pooler) in prod. Switch
  only on `DATABASE_URL`.
- Deploy: Render (`render.yaml`). Its disk is wiped on restart, so nothing durable goes on disk.
- Frontend: static HTML/JS in `static/`, no framework.

## Layout
- `app/main.py` routes, rate limit, `/staff` API, `/health` (deployed commit)
- `app/agent.py` one patient turn: tool registry (Pydantic args), max 6 rounds, guard on the reply
- `app/llm.py` providers over a neutral history format; mid-turn failover, budget, usage tracking
- `app/prompt.py` static system prompt (cacheable); the clock rides on each user turn
- `app/guard.py` unbacked-claim and alert-claim checks, red flags, medical advice, log redaction
- `app/tools.py` the only code that reads/writes patient data for the agent
- `app/dates.py` patient phrases → date windows; DOB extraction
- `app/db.py` models, engine, additive migrations · `app/seed.py` seed + `reset_demo()`
- `evals/` YAML scenarios + runner · `scripts/` smoke test, model check, contention check
- `tests/` unit tests (`make test`)

## Safety principles
- **Unbacked claims:** booking/change claims and "I've told the team" are checked against the DB by
  `app/guard.py`; so is "that time has passed/is taken" (only a tool may say it). The eval runner
  calls the same functions; don't fork the rules.
- **Identity:** name + DOB verification before revealing or changing anything, and `verify_patient`
  accepts only a DOB the patient typed this session (DOB provenance).
- **Emergencies:** `notify_staff` works before registration. `triage_emergency` (code) routes any red
  flag to the 911 text, writes the alert summary, and gates emergency slots. Screening never deadlocks.
  A red-flag turn makes no further LLM call: the reply is the code-written 911 text only.
- **Changes:** reschedule/cancel need a read-back and a "yes" on a later turn. Bookings don't.
- `book` is the only judge of a slot (offered this session, future, open, in hours). A tapped quick
  reply sends its slot ids; the agent tells the model to book exactly those.
- Slot claims are `UPDATE … WHERE status='open'`; bookings are idempotent per session, turn and request.

## Evals and tests
- `expect:` checks are **safety** and gate CI. `style:` checks and tone are reported, never gating.
- Tests and evals pin `DATABASE_URL` to SQLite. Never point tests at a non-local DB: the fixture
  drops tables. Use `scripts/contention_check.py` (throwaway schema) for a remote DB.

## Rules
- Session state lives in the `conversations` table, never in process memory.
- Tools raise `ToolError(code, message)`; the agent turns it into a tool result.
- Never log phone/DOB unmasked. Never let the LLM write SQL. Never commit secrets or `*.db`.
