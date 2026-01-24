from django import template
from django.utils.safestring import mark_safe
import json

register = template.Library()


@register.filter
def json_get(value, key):
    """
    Get a value from a JSON string or dict by key
    """
    if not value:
        return ""

    try:
        if isinstance(value, str):
            data = json.loads(value)
        else:
            data = value

        if isinstance(data, dict):
            return data.get(key, "")
        return ""
    except (json.JSONDecodeError, TypeError, KeyError):
        return ""


@register.filter
def to_json(value):
    """
    Convert a Python object to a JSON string for use in JavaScript.
    Properly handles lists, dicts, and None values.
    """
    if value is None:
        return mark_safe("[]")
    try:
        return mark_safe(json.dumps(value))
    except (TypeError, ValueError):
        return mark_safe("[]")
