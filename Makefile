.PHONY: dev test eval eval-fallback eval-judge

dev:
	uv run uvicorn app.main:app --reload

test:
	uv run pytest -q

# Full scenario suite on EVAL_PROVIDER (default claude); prints the pass/fail table.
eval:
	uv run python -m evals.run

# The 4-scenario subset on the Gemini fallback (throttled for the free tier).
eval-fallback:
	uv run python -m evals.run --smoke-fallback

# Full suite plus an LLM-judge tone score (reported, never gating).
eval-judge:
	uv run python -m evals.run --judge
