"""
Unit tests for the Phase 1 "Crash, Auth, Cost-control & Config quick wins".

All tests are DB-free Django SimpleTestCases (no Postgres / OpenAI required).
They cover the surgical fixes made on the data-insights-fixes branch:

1. Crash fix: MessageStatusSerializer no longer references the non-existent
   ``processing_time_ms`` field (would raise ImproperlyConfigured).
2. Cost control: per-LLM-call output cap (AGENT_MAX_TOKENS) and LangGraph
   tool-loop depth cap (RECURSION_LIMIT) are sourced from settings.
3. Auth / throttling: AIInsightRateThrottle is wired onto ChatSessionView and
   its "ai_insight" scope resolves to a configured rate.
4. Config: SECRET_KEY/DEBUG handling and removal of dead CORS settings.
"""

from django.conf import settings
from django.test import SimpleTestCase


class MessageStatusSerializerTests(SimpleTestCase):
    """Phase 1 crash fix: serializer must not reference a non-existent field."""

    def test_serializer_instantiates_without_error(self):
        """Building the field map must not raise ImproperlyConfigured.

        Accessing ``.fields`` forces DRF to resolve every Meta field against
        the model; the old config listed ``processing_time_ms`` (not a model
        or declared field) and would raise here.
        """
        from data_insights.serializers import MessageStatusSerializer

        serializer = MessageStatusSerializer()
        # Force field resolution explicitly.
        fields = set(serializer.fields.keys())
        self.assertEqual(fields, {"id", "status"})

    def test_processing_time_ms_not_serialized(self):
        from data_insights.serializers import MessageStatusSerializer

        self.assertNotIn("processing_time_ms", MessageStatusSerializer().fields)

    def test_processing_time_ms_not_a_model_field(self):
        """Guards the rationale: the field genuinely does not exist on the model,
        so it cannot be (re)added to the serializer without a migration."""
        from data_insights.models import ChatMessage

        model_fields = {f.name for f in ChatMessage._meta.get_fields()}
        self.assertNotIn("processing_time_ms", model_fields)


class CostControlSettingsTests(SimpleTestCase):
    """Phase 1 cost control: token + recursion caps are configured and sane."""

    def test_max_tokens_in_settings(self):
        self.assertIn("MAX_TOKENS", settings.AI_INSIGHT["AGENT"])
        self.assertEqual(settings.AI_INSIGHT["AGENT"]["MAX_TOKENS"], 2000)

    def test_recursion_limit_in_settings(self):
        self.assertIn("RECURSION_LIMIT", settings.AI_INSIGHT["AGENT"])
        self.assertEqual(settings.AI_INSIGHT["AGENT"]["RECURSION_LIMIT"], 15)

    def test_view_agent_max_tokens_matches_settings(self):
        from data_insights import views

        self.assertEqual(
            views.AGENT_MAX_TOKENS, settings.AI_INSIGHT["AGENT"]["MAX_TOKENS"]
        )

    def test_workflow_recursion_limit_matches_settings(self):
        from data_insights.workflows import agent_workflow

        self.assertEqual(
            agent_workflow.RECURSION_LIMIT,
            settings.AI_INSIGHT["AGENT"]["RECURSION_LIMIT"],
        )

    def test_recursion_limit_is_positive_int(self):
        from data_insights.workflows import agent_workflow

        self.assertIsInstance(agent_workflow.RECURSION_LIMIT, int)
        self.assertGreater(agent_workflow.RECURSION_LIMIT, 0)


class ThrottlingWiringTests(SimpleTestCase):
    """Phase 1 cost control / auth: the AI throttle is actually wired in."""

    def test_throttle_scope(self):
        from data_insights.views import AIInsightRateThrottle

        self.assertEqual(AIInsightRateThrottle.scope, "ai_insight")

    def test_throttle_attached_to_chat_session_view(self):
        from data_insights.views import ChatSessionView, AIInsightRateThrottle

        self.assertIn(AIInsightRateThrottle, ChatSessionView.throttle_classes)

    def test_throttle_rate_configured(self):
        rates = settings.REST_FRAMEWORK.get("DEFAULT_THROTTLE_RATES", {})
        self.assertIn("ai_insight", rates)

    def test_throttle_rate_parses(self):
        """The configured rate string must be parseable by DRF
        (``<num>/<period>``), otherwise requests would error at runtime."""
        from data_insights.views import AIInsightRateThrottle

        num_requests, duration = AIInsightRateThrottle().parse_rate(
            settings.REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]["ai_insight"]
        )
        self.assertIsNotNone(num_requests)
        self.assertGreater(num_requests, 0)
        self.assertEqual(duration, 60)  # "/min" -> 60 seconds


class ConfigHardeningTests(SimpleTestCase):
    """Phase 1 config quick wins: SECRET_KEY and dead CORS config."""

    def test_secret_key_present(self):
        """check / setup already succeeded, so SECRET_KEY resolved to a value
        (real env var, or the labeled dev fallback under DEBUG)."""
        self.assertTrue(settings.SECRET_KEY)

    def test_dead_cors_settings_removed(self):
        """django-cors-headers is not installed and CorsMiddleware is not in
        MIDDLEWARE, so these settings were a no-op and should be gone."""
        self.assertFalse(hasattr(settings, "CORS_ALLOWED_ORIGINS"))
        self.assertFalse(hasattr(settings, "CORS_ALLOW_CREDENTIALS"))

    def test_cors_not_installed(self):
        """Sanity: confirms the removal rationale still holds."""
        self.assertNotIn("corsheaders", settings.INSTALLED_APPS)
        self.assertFalse(
            any("cors" in m.lower() for m in settings.MIDDLEWARE),
            "CorsMiddleware unexpectedly present in MIDDLEWARE",
        )

    def test_production_cookie_hardening_gated_on_debug(self):
        """When DEBUG is False the secure-cookie / SSL settings must be on;
        when DEBUG is True they should not be forced (local HTTP dev)."""
        if settings.DEBUG:
            # Dev: hardening not forced on.
            self.assertFalse(getattr(settings, "SECURE_SSL_REDIRECT", False))
        else:
            self.assertTrue(settings.SESSION_COOKIE_SECURE)
            self.assertTrue(settings.CSRF_COOKIE_SECURE)
            self.assertTrue(settings.SECURE_SSL_REDIRECT)
            self.assertGreater(settings.SECURE_HSTS_SECONDS, 0)


class UnifiedChatAuthTests(SimpleTestCase):
    """Phase 1 auth: the unified_chat view is login-protected."""

    def test_unified_chat_requires_login(self):
        """The view is wrapped by @login_required; calling it with an
        anonymous request must redirect (302) to the login URL rather than
        rendering the page."""
        from django.contrib.auth.models import AnonymousUser
        from django.test import RequestFactory
        from data_insights.views import unified_chat

        request = RequestFactory().get("/data-insights/")
        request.user = AnonymousUser()
        response = unified_chat(request)
        self.assertEqual(response.status_code, 302)
        self.assertIn(settings.LOGIN_URL, response.url)
