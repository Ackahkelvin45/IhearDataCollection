from django import template
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
