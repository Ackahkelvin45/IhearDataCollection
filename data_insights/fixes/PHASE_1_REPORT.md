# Phase 1 — Crash, Auth, Cost-control & Config quick wins

# Phase 1 — Crash, Auth, Cost-control & Config Quick Wins — Phase Report

Branch: `data-insights-fixes` (working tree). App under review: `data_insights/`.
All changes are small, surgical, and additive/one-line where possible. No secrets rotated, no git history rewritten, no commits/pushes performed (owner commits). Cross-collector (org-wide) data visibility was treated as intentional — no per-user data scoping was added anywhere.

Prior-stage gates: Critique = approve (0 must_fix); Testing check_passed = true (0 regressions); Validation acceptance_met = true. A dedicated DB-free test module was added and all 17 tests pass locally (`./venv/bin/python manage.py test data_insights.tests.test_phase1_fixes`).

## 1. Issues fixed (with final status)

| # | Category | Issue | Fix | Status |
|---|----------|-------|-----|--------|
| 1 | Crash | `MessageStatusSerializer` listed `processing_time_ms`, which is neither a model field nor a declared field → DRF raises `ImproperlyConfigured` when the field map resolves. | Removed `processing_time_ms` from `Meta.fields`; left an explanatory comment that processing time is computed transiently in the streaming response and never persisted. | FIXED |
| 2 | Auth | `unified_chat` HTML view rendered the chat page to anonymous users (no auth gate). | Added `@login_required`; anonymous requests now 302-redirect to `LOGIN_URL` (`/auth/login/`). | FIXED |
| 3 | Auth / Cost | AI chat/message endpoints had no rate ceiling; the configured `AI_INSIGHT["SECURITY"]["RATE_LIMIT_PER_MINUTE"]` was dead config. | Added `AIInsightRateThrottle` (scope `ai_insight`), wired onto `ChatSessionView.throttle_classes`. The LLM-invoking `messages` POST action inherits it. Rate sourced from `DEFAULT_THROTTLE_RATES["ai_insight"]` (env `AI_INSIGHT_RATE_LIMIT_PER_MINUTE`, default 30/min/user). | FIXED |
| 4 | Cost control | Unbounded LLM output tokens per call. | Added `MAX_TOKENS` (default 2000) to `AI_INSIGHT["AGENT"]`; applied `max_tokens` to the streaming agent LLM and non-streaming dashboard LLM in `views.py` and to the SQL-agent LLM in `tools.py`. (The 200-token title-generation `ChatOpenAI` was intentionally left alone.) | FIXED |
| 5 | Cost control | Runaway LangGraph tool-loop could rack up unbounded LLM spend. | Added `RECURSION_LIMIT` (default 15) to `AI_INSIGHT["AGENT"]`; passed `recursion_limit` into all four `DataAgent` run configs and into the SQL agent's `invoke` config. | FIXED |
| 6 | Config | `SECRET_KEY` could resolve to `None` and silently boot. | Moved `DEBUG` above `SECRET_KEY`; fail-fast (`ImproperlyConfigured`) when unset in non-DEBUG; clearly-labeled insecure dev fallback under DEBUG. Secret never logged/printed. | FIXED |
| 7 | Config | Dead CORS config: `CORS_ALLOWED_ORIGINS` / `CORS_ALLOW_CREDENTIALS` set despite `django-cors-headers` not installed and `CorsMiddleware` absent from `MIDDLEWARE` (no-op). | Removed both settings; left a comment with the exact 4-step re-enable recipe for a future separate frontend. | FIXED |
| 8 | Config (hardening) | No transport/cookie hardening in production. | Added a `if not DEBUG:` block: `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`, `SECURE_SSL_REDIRECT`, HSTS (1 year, subdomains, preload). Gated on `not DEBUG` so local HTTP dev is unaffected. | FIXED |

## 2. Files modified

- `data_insights/serializers.py` — drop non-existent `processing_time_ms` from `MessageStatusSerializer`.
- `data_insights/views.py` — `@login_required` on `unified_chat`; new `AIInsightRateThrottle`; `AGENT_MAX_TOKENS` constant; `throttle_classes` on `ChatSessionView`; `max_tokens` on the two request-time `ChatOpenAI` instances.
- `data_insights/workflows/agent_workflow.py` — module-level `RECURSION_LIMIT` from settings; injected into the four run-config dicts.
- `data_insights/workflows/tools.py` — `max_tokens` on the cached SQL-agent LLM (`_get_llm`); `recursion_limit` config on `_invoke_sql_agent`'s `agent.invoke`.
- `datacollection/settings.py` — `DEBUG` reordered before `SECRET_KEY` with fail-fast/dev-fallback; `DEFAULT_THROTTLE_RATES["ai_insight"]`; removed dead CORS settings (replaced with a re-enable comment); `if not DEBUG` cookie/SSL/HSTS hardening; `MAX_TOKENS` + `RECURSION_LIMIT` added to both the `USE_SQLITE` and Docker branches of `AI_INSIGHT["AGENT"]`.
- `data_insights/tests/test_phase1_fixes.py` — NEW. 17 DB-free `SimpleTestCase`s covering all of the above (no Postgres/OpenAI required).

## 3. Key decisions & tradeoffs

- All new behavior is config-driven via `AI_INSIGHT` / env vars with backward-compatible defaults (`MAX_TOKENS=2000`, `RECURSION_LIMIT=15`, throttle `30/min`), keeping the changes additive and tunable without code edits.
- Throttle/cost caps deliberately set generous so existing clients are not broken while still bounding worst-case LLM spend.
- The serializer crash was fixed by removing the field rather than adding it to the model, since processing time is transient (computed in the streaming response) and persisting it would require a migration — out of scope for a quick win.
- Dead CORS config was removed rather than "fixed," because the middleware/package backing it does not exist; a precise re-enable recipe was left in-place so intent is not lost.
- Production hardening is uniformly gated on `not DEBUG` so the local HTTP dev workflow is untouched.
- Per the product decision, no per-user data scoping was introduced; throttling is per-user only as a spend/abuse ceiling, not a visibility boundary.
- The duplicated `AI_INSIGHT["AGENT"]` block (SQLite vs Docker branch) was updated in both places to preserve existing structure rather than refactoring the config into a single source — lower risk for a quick-win phase.

## 4. Risks introduced

- LOW — `max_tokens=2000` could truncate an unusually long single LLM response. Default chosen to comfortably exceed normal answers; tunable via `AI_INSIGHT_MAX_TOKENS`.
- LOW — `recursion_limit=15` could cut off a legitimately deep multi-tool reasoning chain, surfacing as a LangGraph recursion error. Tunable via `AI_INSIGHT_RECURSION_LIMIT`.
- LOW — `30/min` throttle could 429 a heavy legitimate user. Tunable via `AI_INSIGHT_RATE_LIMIT_PER_MINUTE`.
- LOW/OPERATIONAL — Non-DEBUG boot now hard-fails without `SECRET_KEY`. Intended fail-fast; requires `.env`/env to be present in any non-debug environment (already the expectation).
- LOW/OPERATIONAL — `SECURE_SSL_REDIRECT` + HSTS in production assume TLS terminates correctly upstream (project already sets `SECURE_PROXY_SSL_HEADER`/`USE_X_FORWARDED_HOST`, so this is consistent). A misconfigured proxy could cause redirect loops — standard caveat.

## 5. Remaining concerns

- The dev `SECRET_KEY` fallback only triggers under DEBUG, but anyone running DEBUG=True shares the same hardcoded labeled key — acceptable for local dev, must never be used in any shared/staging-with-debug environment.
- `AI_INSIGHT["AGENT"]` is duplicated across the two settings branches; the two copies must be kept in sync manually (future drift risk).
- Throttling is in-memory by default (DRF's default cache). In a multi-process/multi-worker deployment the effective limit is per-process unless a shared cache backend is configured for throttling.
- The previously-listed `AI_INSIGHT["SECURITY"]["RATE_LIMIT_PER_MINUTE"]` and the new `DEFAULT_THROTTLE_RATES["ai_insight"]` both read the same env var but are now two declarations of the same intent; consider consolidating so they cannot diverge.

## 6. Recommended follow-up work

1. Add Sentry/error monitoring or at least structured logging around throttle 429s and LangGraph recursion-limit hits to detect when the caps bite real users.
2. Configure a shared cache backend for DRF throttling so limits are enforced cluster-wide, not per-process.
3. Consolidate the duplicated `AI_INSIGHT["AGENT"]` config (and the dual rate-limit declarations) into a single source of truth.
4. If processing-time-per-message is a desired product metric, add a real `processing_time_ms` model field + migration and re-introduce it to the serializer.
5. Decide whether CORS is actually needed (separate frontend). If yes, follow the in-file recipe; if permanently not, remove the leftover `CSRF_TRUSTED_ORIGINS` cruft review too.
6. Consider per-token/cost accounting (not just per-call token caps) for true spend budgeting across a session.
7. Run the broader `data_insights` test suite and a smoke test against a live OpenAI key in staging to confirm `max_tokens`/`recursion_limit` defaults don't degrade real answers.

## Backlog (next phases)

- P1: Configure a shared cache backend (e.g. Redis) for DRF throttling so the ai_insight rate is enforced cluster-wide across workers.
- P1: Add monitoring/structured logging for 429 throttle responses and LangGraph recursion-limit errors to catch when caps truncate legitimate usage; revisit MAX_TOKENS/RECURSION_LIMIT defaults from real data.
- P2: Consolidate the duplicated AI_INSIGHT['AGENT'] config block and the dual rate-limit declarations into a single source of truth to prevent drift.
- P2: Run the full data_insights test suite plus a live-key staging smoke test to confirm max_tokens=2000 / recursion_limit=15 do not degrade real answers.
- P2: Decide CORS direction — either install django-cors-headers + middleware per the in-file recipe for a separate frontend, or fully clean up remaining CSRF_TRUSTED_ORIGINS / CORS cruft.
- P3: If processing-time-per-message is a product metric, add a real processing_time_ms model field + migration and re-add it to MessageStatusSerializer.
- P3: Add per-session/per-user token-and-cost accounting (beyond per-call token caps) for true LLM spend budgeting.
- P3: Replace the DEBUG hardcoded dev SECRET_KEY fallback with a generated-per-environment dev key, or document that DEBUG must never be enabled in shared environments.
