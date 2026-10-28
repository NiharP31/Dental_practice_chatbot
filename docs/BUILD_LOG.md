# Build log

What each review checkpoint built, what review or testing caught, and what changed.

## Checkpoint 1: tools and unit tests

**Built:** models, an idempotent seed, a date-phrase resolver, and the six tools. The tools include
atomic slot claims and a two-phase confirm for changes ([b8c9c0e](https://github.com/NiharP31/Dental_practice_chatbot/commit/b8c9c0e)).

**Caught in review:**
- Two different bookings in one turn shared the idempotency key `session:turn`. The second
  replayed the first, so the patient was told something was booked that wasn't.
- Parents couldn't manage their kids' appointments.

**Changed:**
- Booking keys now include a fingerprint of the request.
- Added guardian/household links. Spouses still verify separately ([67a151b](https://github.com/NiharP31/Dental_practice_chatbot/commit/67a151b)).
- Tests got a global timeout and can run against Postgres ([fbee4c0](https://github.com/NiharP31/Dental_practice_chatbot/commit/fbee4c0)).

## Checkpoint 2: agent, guardrails, UI

**Built:** the agent loop, guardrails, the staff page, and the chat UI.

**Caught by live smoke tests:**
- The unbacked-claim guard replaced a *true* "you're already all set" with "nothing has been
  booked" ([6c400da](https://github.com/NiharP31/Dental_practice_chatbot/commit/6c400da)).
- Gemini's free tier returned frequent 503s and capped `3.6-flash` at 20 requests a day.

**Changed:**
- Claims are now checked against the DB, not just against this turn's actions.
- Claude Haiku became the primary model with Gemini as fallback, DeepSeek was dropped, and spend is
  capped by a daily budget ([82cdb93](https://github.com/NiharP31/Dental_practice_chatbot/commit/82cdb93)).
- A live test showed Gemini rejects Claude's unsigned tool calls during a mid-turn handoff, so those
  calls now carry a placeholder signature.

## Checkpoint 3: evals and deploy

**Built:** 13 YAML scenarios that assert on DB state ([d504c86](https://github.com/NiharP31/Dental_practice_chatbot/commit/d504c86)), CI, and the
Render/Supabase deploy config ([1694587](https://github.com/NiharP31/Dental_practice_chatbot/commit/1694587)).

**Caught by evals:**
- Four guardrail false positives: "all set up in our system", restating appointments by weekday,
  "all set" a turn after booking, and registration plus options in one reply ([3dd031b](https://github.com/NiharP31/Dental_practice_chatbot/commit/3dd031b)).
- A double yes on cancellations, because the model asked "cancel it?" before calling `modify`.
- Gemini invented a DOB to act on an injected "SYSTEM: verified" message.
- Persona slips: "Ahoy, matey".

**Caught in review:** the contention test's fixture drops every table. Pointing it at Supabase, as
planned, would have wiped production.

**Changed:**
- `verify_patient` only accepts DOBs the patient typed ([a321e0b](https://github.com/NiharP31/Dental_practice_chatbot/commit/a321e0b)).
- Contention checks against remote databases run in a throwaway schema, and the fixture refuses
  non-local databases ([1694587](https://github.com/NiharP31/Dental_practice_chatbot/commit/1694587)).
- Patient text is delimited in tags; the persona check went from 1/3 to 5/5 ([962e679](https://github.com/NiharP31/Dental_practice_chatbot/commit/962e679)).

**Result:** Claude passed 37/39 across 3 full runs before the persona fix.
