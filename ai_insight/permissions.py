from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from datetime import timedelta
from rest_framework.permissions import BasePermission
from rest_framework.exceptions import Throttled


class CanUseAIInsight(BasePermission):
    """
    Permission to check if user can use AI insight features
    Includes rate limiting and session management
    """

    message = "User does not have permission to use AI insight agent."

    def has_permission(self, request, view):  # type: ignore[override]
        if not (
            request.user
            and request.user.is_authenticated
            and request.user.has_perm("ai_insight.use_ai_insight")
        ):
            return False

        if not self._check_rate_limit(request):
            return False

        if view.action == "create" and hasattr(view, "get_serializer_class"):
            pass

        return True

    def has_object_permission(self, request, view, obj):  # type: ignore[override]
        if hasattr(obj, "user"):
            return obj.user == request.user
        elif hasattr(obj, "session"):
            return obj.session.user == request.user
        return False

    def _check_rate_limit(self, request) -> bool:
        """Check if user has exceeded rate limits"""
        ai_config = getattr(settings, "AI_INSIGHT", {})
        security_config = ai_config.get("SECURITY", {})
        rate_limit = security_config.get("RATE_LIMIT_PER_MINUTE", 30)

        if rate_limit <= 0:
            return True

        user_id = request.user.id
        cache_key = f"ai_insight_rate_limit:{user_id}"
        current_time = timezone.now()

        requests_data = cache.get(cache_key, [])

        one_minute_ago = current_time - timedelta(minutes=1)
        recent_requests = [
            req_time
            for req_time in requests_data
            if req_time > one_minute_ago.timestamp()
        ]

        if len(recent_requests) >= rate_limit:
            raise Throttled(
                detail=f"Rate limit exceeded. Maximum {rate_limit} requests per minute.",
                wait=60,
            )

        recent_requests.append(current_time.timestamp())
        cache.set(cache_key, recent_requests, timeout=120)

        return True


class CanManageAISessions(BasePermission):
    """
    Permission for users who can manage AI sessions (admins)
    """

    message = "User does not have permission to manage AI insight sessions."

    def has_permission(self, request, view):  # type: ignore[override]
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.has_perm("ai_insight.manage_ai_sessions")
        )


class IsSessionOwnerOrReadOnly(BasePermission):
    def has_permission(self, request, view):  # type: ignore[override]
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.has_perm("ai_insight.use_ai_insight")
        )

    def has_object_permission(self, request, view, obj):  # type: ignore[override]
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            return True

        if hasattr(obj, "user"):
            return obj.user == request.user
        elif hasattr(obj, "session"):
            return obj.session.user == request.user

        return False
