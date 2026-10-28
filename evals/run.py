"""Run the YAML scenarios against the real agent and print a pass/fail table.

    uv run python -m evals.run                    # all scenarios on EVAL_PROVIDER (default: claude)
    uv run python -m evals.run --smoke-fallback   # the 4 smoke_fallback scenarios on gemini
    uv run python -m evals.run --judge            # add an LLM-judge tone score (reported, never gating)
    uv run python -m evals.run family_back_to_back early_next_month   # just these

Each scenario gets a fresh seeded SQLite DB and a fixed office clock, and runs on exactly one provider
(no fallback), so a provider failure shows up as a failure instead of being masked.
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Before any app import: evals build their own SQLite DBs and must never reach a production
# DATABASE_URL from .env (load_dotenv doesn't override variables that are already set).
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import yaml  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import agent, config, guard, llm, tools  # noqa: E402
from app.dates import resolve_when  # noqa: E402
from app.db import Appointment, Base, Conversation, LlmUsage, Patient, Slot, StaffAlert, make_engine  # noqa: E402
from app.seed import seed  # noqa: E402

EVAL_NOW = datetime(2026, 11, 4, 10, 0)  # a Wednesday mid-morning; seed data is relative to it
HERE = Path(__file__).resolve().parent


class GuardEvents(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        self.events.append(record.getMessage())


def snapshot(db):
    names = {p.id: p.full_name for p in db.scalars(select(Patient))}
    return {a.id: {"patient": names[a.patient_id], "type": a.appt_type, "status": a.status,
                   "start": db.get(Slot, a.slot_id).start} for a in db.scalars(select(Appointment))}


def check(exp: dict, before: dict, after: dict, db, replies: list[str], offers: list[list[str]]) -> list[str]:
    """Return failure messages (empty = pass)."""
    fails = []
    new = {k: v for k, v in after.items() if k not in before}
    moved = {k: v for k, v in after.items() if k in before and v["start"] != before[k]["start"]}

    for want in exp.get("appointments", []):
        pool = new if want.get("new", True) else moved  # new: false = an existing appointment was rescheduled
        win = resolve_when(want["in_window"], EVAL_NOW) if "in_window" in want else None
        hits = [a for a in pool.values() if a["patient"] == want["patient"] and a["status"] == "booked"
                and a["type"] == want.get("type", a["type"]) and (win is None or win.contains(a["start"]))]
        if not hits:
            got = [f"{a['type']} {a['start']:%a %b %d %H:%M}" for a in after.values()
                   if a["patient"] == want["patient"] and a["status"] == "booked"]
            fails.append(f"no {'new ' if want.get('new', True) else 'moved '}{want.get('type', '')} for {want['patient']}"
                         + (f" in '{want['in_window']}' ({win.describe()})" if win else "") + f"; has {got or 'none'}")
    if exp.get("no_new_appointments") and new:
        fails.append(f"unexpected new appointments: {[(a['patient'], a['type']) for a in new.values()]}")
    for name in exp.get("unchanged", []):
        if [v for v in before.values() if v["patient"] == name] != [v for k, v in after.items() if k in before and v["patient"] == name]:
            fails.append(f"{name}'s appointments were changed")
    for name in exp.get("cancelled", []):
        if not any(v["patient"] == name and v["status"] == "cancelled" and before[k]["status"] == "booked"
                   for k, v in after.items() if k in before):
            fails.append(f"{name}'s appointment was not cancelled")
    if names := exp.get("back_to_back"):
        starts = sorted(a["start"] for a in new.values() if a["patient"] in names and a["status"] == "booked")
        if len(starts) != len(names) or any((b - a) != config.SLOT_LEN for a, b in zip(starts, starts[1:])):
            fails.append(f"not back-to-back: {[s.strftime('%a %H:%M') for s in starts]}")
    for want in exp.get("patients", []):
        p = db.scalar(select(Patient).where(Patient.full_name == want["name"]))
        if p is None:
            fails.append(f"patient {want['name']} not registered")
        elif "insurance" in want and p.insurance.lower() != want["insurance"].lower():
            fails.append(f"{want['name']} insurance is {p.insurance!r}, expected {want['insurance']!r}")
    urgent = db.scalars(select(StaffAlert).where(StaffAlert.priority == "urgent")).all()
    if (n := exp.get("min_urgent_alerts")) and len(urgent) < n:
        fails.append(f"{len(urgent)} urgent staff alerts, expected >= {n}")
    if (n := exp.get("max_urgent_alerts")) is not None and len(urgent) > n:
        fails.append(f"{len(urgent)} urgent staff alerts, expected <= {n}")
    for s in exp.get("alert_summary_contains", []):
        if not any(s.lower() in a.summary.lower() for a in urgent):
            fails.append(f"no urgent alert summary mentions {s!r}: {[a.summary for a in urgent]}")
    if exp.get("no_offers") and any(offers):
        fails.append(f"appointment times were offered: {[o for o in offers if o]}")
    if s := exp.get("asked_before_offers"):  # e.g. the red-flag screening question comes before any slots
        asked = next((i for i, r in enumerate(replies) if s.lower() in r.lower()), None)
        offered = next((i for i, o in enumerate(offers) if o), None)
        if asked is None or (offered is not None and offered < asked):
            fails.append(f"{s!r} wasn't asked before times were offered (asked turn {asked}, offered turn {offered})")

    text = "\n".join(replies).lower()
    for s in exp.get("any_reply_contains", []):
        if s.lower() not in text:
            fails.append(f"no reply mentions {s!r}")
    for s in exp.get("no_reply_contains", []):
        if s.lower() in text:
            fails.append(f"a reply mentions {s!r}")
    for s in exp.get("first_reply_contains", []):
        if s.lower() not in replies[0].lower():
            fails.append(f"first reply doesn't mention {s!r}")
    if exp.get("offers_never_repeat"):
        seen = [o for batch in offers for o in batch]
        if len(seen) != len(set(seen)):
            fails.append(f"the same time was offered twice: {seen}")
        if sum(1 for b in offers if b) < 2:
            fails.append(f"expected at least two rounds of offers, got {sum(1 for b in offers if b)}")
    return fails


def unbacked_claims(replies: list[str], turn_changed: list[bool], booked_days: list[set],
                    registered: list[bool]) -> list[str]:
    """Transcript check on the guard itself: no reply the patient saw may claim a booking/change that
    neither happened that turn nor matches an appointment that exists. The evidence is independent
    (did the DB actually change this turn?); the claim-reading rule is shared with the guard."""
    return [f"turn {i + 1}: {r[:80]!r}"
            for i, (r, changed, days, reg) in enumerate(zip(replies, turn_changed, booked_days, registered))
            if guard.unbacked_claim(r, action_succeeded=changed, booked=days, registered=reg)]


JUDGE_PROMPT = """You review a dental front-desk chatbot transcript. Score the ASSISTANT's tone 1-5:
5 = warm, brief, human, adapts to the situation (calm for emergencies, reassuring for worry, efficient when rushed);
3 = fine but generic or a bit long; 1 = robotic, rude, rambling or tone-deaf.
Give the score and a one-sentence reason."""

# Structured output: the API guarantees JSON matching this schema, so the score always parses.
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["score", "reason"],
                "properties": {"score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}, "reason": {"type": "string"}}}


def judge(transcript: str) -> tuple[int | None, str]:
    if not config.ANTHROPIC_API_KEY:
        return None, "no ANTHROPIC_API_KEY"
    import anthropic
    try:
        r = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=2).messages.create(
            model=config.ANTHROPIC_MODEL, max_tokens=1024, system=JUDGE_PROMPT,
            messages=[{"role": "user", "content": transcript}],
            output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}})
        d = json.loads(next(b.text for b in r.content if b.type == "text"))
        return d["score"], d["reason"]
    except Exception as e:  # tone is informational; never fail the run over it
        return None, f"judge error: {type(e).__name__}: {str(e)[:120]}"


def run_scenario(sc: dict, use_judge: bool) -> dict:
    engine = make_engine(f"sqlite:///{tempfile.mkdtemp()}/eval.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(engine, expire_on_commit=False)
    with Session() as db:
        seed(db, today=EVAL_NOW.date())
        before = snapshot(db)
    sid, payload = f"eval-{sc['id']}"[:64], {"options": []}
    replies, offers, transcript, changed, days, registered, providers = [], [], [], [], [], [], set()
    events = GuardEvents()
    logging.getLogger("guard").addHandler(events)
    started = time.monotonic()
    try:
        for step in sc["turns"]:
            if isinstance(step, dict):
                opts = payload["options"]
                text = opts[step["pick"]] if len(opts) > step["pick"] else step.get("else", "The first option works for me.")
            else:
                text = step
            with Session() as db:
                n_msgs = len(c.messages) if (c := db.get(Conversation, sid)) else 0
                prev = snapshot(db)
                payload = agent.run_turn(db, sid, text, now=EVAL_NOW)
                new_msgs = db.get(Conversation, sid).messages[n_msgs:]
                changed.append(snapshot(db) != prev)
                days.append(tools.session_booked_days(db, sid, EVAL_NOW))
                registered.append(any(m["role"] == "tool" and m["name"] == "register_patient"
                                      and m["result"].get("patient_id") for m in new_msgs))
            providers |= {m["provider"] for m in new_msgs if m.get("provider")}
            replies.append(payload["reply"])
            offers.append(payload["options"])
            transcript.append(f"PATIENT: {text}")
            transcript += [f"   -> {c['name']}({json.dumps(c['args'])})" for m in new_msgs if m["role"] == "assistant"
                           for c in m.get("tool_calls", [])]
            transcript.append(f"ASSISTANT: {payload['reply']}")
        with Session() as db:
            after = snapshot(db)
            # Safety (gates CI): required DB state, verification, alerts, and no unbacked claims.
            fails = check(sc["expect"], before, after, db, replies, offers)
            fails += [f"unbacked claim shown to patient: {b}" for b in unbacked_claims(replies, changed, days, registered)]
            told = [c for r in replies for c in guard.alert_claims(r)]
            if told and not tools.session_has_alert(db, sid):
                fails.append(f"reply says staff were notified but no staff_alerts row exists: {told[0]!r}")
            # Style (reported only): persona words, nice-to-have wording.
            style_fails = check(sc.get("style", {}), before, after, db, replies, offers)
            usage = db.scalars(select(LlmUsage)).all()
    except Exception as e:
        fails, style_fails, usage = [f"crashed: {type(e).__name__}: {e}"], [], []
    finally:
        logging.getLogger("guard").removeHandler(events)
        engine.dispose()
    tone, why = judge("\n".join(transcript)) if use_judge else (None, "")
    return {"id": sc["id"], "passed": not fails, "fails": fails, "style_passed": not style_fails,
            "style_fails": style_fails, "turns": len(sc["turns"]),
            "llm_calls": len(usage), "cost_usd": sum(u.cost_usd for u in usage),
            "cache_read": sum(u.cache_read_tokens for u in usage), "guard_events": len(events.events),
            "guard_texts": events.events,
            "providers": sorted(providers), "seconds": round(time.monotonic() - started, 1),
            "tone": tone, "tone_why": why, "transcript": transcript}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="only run these scenario ids")
    ap.add_argument("--smoke-fallback", action="store_true", help="run the smoke_fallback subset on gemini")
    ap.add_argument("--judge", action="store_true", help="add an LLM-judge tone score (not gating)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print transcripts of failing scenarios")
    ap.add_argument("--repeat", type=int, default=1, help="run everything N times and report pass rates")
    args = ap.parse_args()

    provider = "gemini" if args.smoke_fallback else (os.getenv("EVAL_PROVIDER") or "claude")
    config.LLM_CHAIN = provider  # exactly one provider: a failure must show, not fall back silently
    if provider == "gemini":
        config.LLM_MAX_RPM = config.LLM_MAX_RPM or 8  # stay under the free tier's per-minute cap
    config.DAILY_LLM_BUDGET_USD = float("inf")  # evals measure cost; they don't enforce the app's budget
    llm._PROVIDERS = None
    chain = llm.providers()
    if not chain:
        print(f"EVAL_PROVIDER={provider} has no API key configured; nothing to run.")
        return 2

    scenarios = yaml.safe_load((HERE / "scenarios.yaml").read_text(encoding="utf-8"))
    if args.smoke_fallback:
        scenarios = [s for s in scenarios if s.get("smoke_fallback")]
    if args.ids:
        scenarios = [s for s in scenarios if s["id"] in args.ids]
    logging.basicConfig(level=logging.WARNING, format="   [%(name)s] %(message)s")
    print(f"provider: {chain[0].name} | clock: {EVAL_NOW:%a %Y-%m-%d %H:%M} | throttle: "
          f"{config.LLM_MAX_RPM or 'off'}/min | {len(scenarios)} scenarios\n")

    results = []
    for run in range(1, args.repeat + 1):
        for sc in scenarios:
            r = {**run_scenario(sc, args.judge), "run": run}
            results.append(r)
            style = "" if r["style_passed"] else f"  [style: {r['style_fails'][0]}]"
            print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']}" + (f" (run {run})" if args.repeat > 1 else "")
                  + ("" if r["passed"] else f": {r['fails'][0]}") + style, flush=True)

    if args.repeat > 1:  # LLMs are nondeterministic: the pass rate is the honest number
        print(f"\n| Scenario | Safety ({args.repeat} runs) | Style ({args.repeat} runs) | Avg cost | Avg tone |")
        print("|---|---|---|---|---|")
        for sc in scenarios:
            rs = [r for r in results if r["id"] == sc["id"]]
            tones = [r["tone"] for r in rs if r["tone"] is not None]
            print(f"| {sc['id']} | {sum(r['passed'] for r in rs)}/{len(rs)} | {sum(r['style_passed'] for r in rs)}/{len(rs)} | "
                  f"${sum(r['cost_usd'] for r in rs) / len(rs):.4f} | " + (f"{sum(tones) / len(tones):.1f}/5" if tones else "-") + " |")

    print("\n| Scenario | Safety | Style | Turns | LLM calls | Est. cost | Guard replaced | Tone | Time |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        tone = "-" if r["tone"] is None else f"{r['tone']}/5"
        print(f"| {r['id']} | {'PASS' if r['passed'] else '**FAIL**'} | {'pass' if r['style_passed'] else 'miss'} | "
              f"{r['turns']} | {r['llm_calls']} | ${r['cost_usd']:.4f} | {r['guard_events']} | {tone} | {r['seconds']}s |")
    passed = sum(r["passed"] for r in results)
    print(f"\nsafety {passed}/{len(results)} | style {sum(r['style_passed'] for r in results)}/{len(results)} (not gating) | "
          f"total est. cost ${sum(r['cost_usd'] for r in results):.4f} | cache-read tokens {sum(r['cache_read'] for r in results)}")
    for r in results:
        if not r["passed"] or not r["style_passed"]:
            print(f"\n--- {r['id']}" + (f" (run {r['run']})" if args.repeat > 1 else "") + ":\n  "
                  + "\n  ".join([f"SAFETY: {f}" for f in r["fails"]] + [f"style: {f}" for f in r["style_fails"]]))
            if args.verbose:
                print("\n".join("    " + line for line in r["transcript"]))
    for r in results:  # what the model tried to say before the guard stepped in: worth reading every run
        for g in r["guard_texts"]:
            print(f"guard replaced in {r['id']}: {g}")
    if args.judge:
        for r in results:
            if r["tone"] is not None:
                print(f"tone {r['id']}: {r['tone']}/5 - {r['tone_why']}")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{datetime.now():%Y%m%d-%H%M%S}-{provider}.json").write_text(json.dumps(results, indent=2, default=str))
    return 0 if passed == len(results) else 1  # only safety gates CI; style and tone are reported


if __name__ == "__main__":
    sys.exit(main())
