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
- **Postgres parity:** `docker compose up -d` then
  `TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:55432/test uv run pytest`
  runs the suite (incl. the double-booking contention test) on Postgres.
