# Notes for the README (known limitations / production follow-ups)

- **Local-time storage is single-office only.** Slot times are naive office wall-clock times in
  `OFFICE_TZ`. Multi-location: store UTC, give each office its own timezone, convert at the edges.
- **Verification lockout is per session.** 3 failed name+DOB attempts locks that session only; a new
  session starts fresh. Production: rate-limit per patient record (and per IP) and alert staff.
- **Booking idempotency is per (session, turn, request fingerprint).** It protects against a retried
  tool call / retried turn re-issuing the *same* booking. A retried turn in which the model picks
  *different* slots would still be a new booking, so `/chat` also dedupes on the client's turn number.
- **Households:** a verified guardian can book/modify for linked dependents. Spouses are not linked
  and verify separately. Links are created at registration (`guardian_id`) or by staff; there's no
  chat flow to link two existing records, on purpose (that needs staff review).
- **LLM choice: Claude first, for reliability.** The deployed demo runs Claude Haiku first, then
  Gemini Flash-Lite, then a canned reply with the office phone. During testing, the free Gemini tier
  returned frequent 503s ("high demand"), and `gemini-3.6-flash` had a 20-requests-per-day free cap,
  which works out to roughly 10 conversations. The app still runs on a free Gemini key alone, since
  providers without a key are skipped. Haiku costs an estimated ~$0.01 per conversation.
- **Cost protection exists because a public URL spends real credits.** Limits: 30 turns per session,
  20 POSTs per minute per IP, and `DAILY_LLM_BUDGET_USD` (default $3). Spend is estimated from token
  usage in `llm_usage`. Once the budget is hit, Claude is skipped for the rest of the office day and
  Gemini answers. The rate limiter is per process (fine for one Render instance; Redis/edge beyond).
- **Bookings vs changes.** A new booking happens as soon as the patient picks a slot, with no extra
  "are you sure?". It's low risk and easy to undo, and an extra step loses patients. Reschedules and
  cancellations need an explicit yes on a later turn, enforced in code, because they remove
  something the patient already has.
- **Claude runs with thinking off (effort `low`).** Claude 5.5 thinking blocks are invalidated (400)
  by any edit to earlier history. Our history must stay editable: the guardrail replaces unbacked
  replies, and a turn can switch provider midway. Thinking off also keeps replies fast. The system
  prompt is fully static (the clock rides on each user turn), so it and the tool definitions are
  prompt-cached.
- **Cross-provider failover inside a turn.** Gemini 3 rejects a current-turn function call that has
  no thought signature (verified live: 400). Claude's calls have none, so they're sent with
  Gemini's `skip_thought_signature_validator` placeholder.
- **Postgres parity:** `docker compose up -d` then
  `TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:55432/test uv run pytest`
  runs the suite (incl. the double-booking contention test) on Postgres.
