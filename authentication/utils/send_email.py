import os
from jinja2 import Environment, FileSystemLoader
import logging
import requests
from typing import Dict
from datetime import datetime
from django.conf import settings

logger = logging.getLogger(__name__)


def generic_send_mail(recipient, title, payload: Dict[str, str] = {}):
    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates"))
    )
    template = env.get_template("generic_email.html")

    # Add logo URL to payload if not already present
    if "logo_url" not in payload:
        # Try to get the domain from settings or use a default
        domain = getattr(settings, "EMAIL_DOMAIN", "https://ihearandsee-at-rail.com")
        if not domain.startswith("http"):
            domain = f"https://{domain}"

        # Construct the logo URL
        logo_path = "assets/img/rail.png"
        if hasattr(settings, "USE_S3") and settings.USE_S3:
            # If using S3, construct the full S3 URL
            logo_url = f"{settings.STATIC_URL}{logo_path}"
        else:
            # For local development, use the domain + static path
            logo_url = f"{domain}/static/{logo_path}"

        payload["logo_url"] = logo_url

    # Add current year if not present
    if "current_year" not in payload:
        payload["current_year"] = datetime.now().year

    html_message = template.render(payload)
    logger.info(f"sending email to {recipient}")
    try:
        base_url = "https://0qmusixj1f.execute-api.us-east-1.amazonaws.com/sendEmail"
        body = {
            "recipient": recipient,
            "subject": title,
            "body": html_message,
        }
        email_send = requests.post(
            base_url, json=body, headers={"Content-Type": "application/json"}
        )
        print(f"== response: {email_send.text}")
        return "Mail Sent"
    except Exception as e:
        logger.warning(f"An error occurred sending email {str(e)}")
        return None
