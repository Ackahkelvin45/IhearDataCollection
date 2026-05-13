"""
Manage Knowledge Base URLs for the chatbot.

Usage:
  python manage.py manage_kb_urls --add "https://rail.knust.edu.gh" --title "RAIL Lab" --desc "RAIL lab info, admissions, research"
  python manage.py manage_kb_urls --list
  python manage.py manage_kb_urls --refresh                          # refresh all active
  python manage.py manage_kb_urls --refresh-url "https://rail.knust.edu.gh"
  python manage.py manage_kb_urls --deactivate "https://rail.knust.edu.gh"
  python manage.py manage_kb_urls --activate "https://rail.knust.edu.gh"
  python manage.py manage_kb_urls --remove "https://rail.knust.edu.gh"
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

User = get_user_model()


class Command(BaseCommand):
    help = "Manage Knowledge Base URLs for the chatbot"

    def add_arguments(self, parser):
        parser.add_argument("--add", type=str, help="URL to add to the knowledge base")
        parser.add_argument(
            "--title", type=str, help="Title for the URL (used with --add)"
        )
        parser.add_argument(
            "--desc", type=str, default="", help="Description (used with --add)"
        )
        parser.add_argument(
            "--refresh-freq",
            type=str,
            default="daily",
            choices=["manual", "hourly", "daily", "weekly"],
            help="Refresh frequency (default: daily)",
        )
        parser.add_argument(
            "--max-chars",
            type=int,
            default=8000,
            help="Max chars per fetch (default: 8000)",
        )
        parser.add_argument("--list", action="store_true", help="List all KB URLs")
        parser.add_argument(
            "--refresh", action="store_true", help="Refresh ALL active KB URLs now"
        )
        parser.add_argument(
            "--refresh-url", type=str, help="Refresh a specific URL now"
        )
        parser.add_argument("--deactivate", type=str, help="Deactivate a URL")
        parser.add_argument("--activate", type=str, help="Activate a URL")
        parser.add_argument("--remove", type=str, help="Remove a URL completely")

    def handle(self, *args, **options):
        from chatbot.models import KnowledgeBaseURL
        from chatbot.tasks import refresh_kb_url_task, refresh_all_active_kb_urls_task

        if options["add"]:
            self._add_url(options)
        elif options["list"]:
            self._list_urls()
        elif options["refresh"]:
            self.stdout.write("Queueing refresh for all active KB URLs...")
            result = refresh_all_active_kb_urls_task()
            self.stdout.write(
                self.style.SUCCESS(f"Queued {result.get('queued', 0)} URLs for refresh")
            )
        elif options["refresh_url"]:
            self._refresh_single(options["refresh_url"])
        elif options["deactivate"]:
            self._set_active(options["deactivate"], False)
        elif options["activate"]:
            self._set_active(options["activate"], True)
        elif options["remove"]:
            self._remove_url(options["remove"])
        else:
            self.stdout.write("No action specified. Use --add, --list, --refresh, etc.")

    def _add_url(self, options):
        from chatbot.models import KnowledgeBaseURL
        from chatbot.tasks import refresh_kb_url_task

        url = options["add"].strip()
        title = options["title"] or url

        user = User.objects.filter(is_superuser=True).first() or User.objects.first()

        kb_url, created = KnowledgeBaseURL.objects.update_or_create(
            url=url,
            defaults={
                "title": title,
                "description": options["desc"],
                "refresh_frequency": options["refresh_freq"],
                "max_chars_per_fetch": options["max_chars"],
                "is_active": True,
                "added_by": user,
            },
        )

        if created:
            self.stdout.write(self.style.SUCCESS(f"Added: {title} ({url})"))
        else:
            self.stdout.write(self.style.WARNING(f"Updated existing: {title} ({url})"))

        # Fetch content immediately
        self.stdout.write("Fetching and indexing content...")
        refresh_kb_url_task.delay(str(kb_url.id))
        self.stdout.write("Refresh queued (runs in background via Celery).")

    def _list_urls(self):
        from chatbot.models import KnowledgeBaseURL

        urls = KnowledgeBaseURL.objects.all().order_by("-created_at")
        if not urls:
            self.stdout.write("No KB URLs configured.")
            return

        self.stdout.write(
            f"{'Status':<8} {'Refresh':<10} {'Chunks':<8} {'Title':<30} URL"
        )
        self.stdout.write("-" * 100)
        for u in urls:
            status = "Active" if u.is_active else "Inactive"
            refresh = u.refresh_frequency
            chunks = str(u.total_chunks)
            last = (
                u.last_fetched_at.strftime("%Y-%m-%d") if u.last_fetched_at else "never"
            )
            self.stdout.write(
                f"{status:<8} {refresh:<10} {chunks:<8} {u.title[:28]:<30} {u.url[:50]}"
            )
            self.stdout.write(f"  Last fetch: {last} | Success: {u.last_fetch_success}")

    def _refresh_single(self, url):
        from chatbot.models import KnowledgeBaseURL
        from chatbot.tasks import refresh_kb_url_task

        try:
            kb_url = KnowledgeBaseURL.objects.get(url=url)
        except KnowledgeBaseURL.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"URL not found: {url}"))
            return

        refresh_kb_url_task.delay(str(kb_url.id))
        self.stdout.write(self.style.SUCCESS(f"Refresh queued for: {kb_url.title}"))

    def _set_active(self, url, active):
        from chatbot.models import KnowledgeBaseURL

        try:
            kb_url = KnowledgeBaseURL.objects.get(url=url)
        except KnowledgeBaseURL.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"URL not found: {url}"))
            return

        kb_url.is_active = active
        kb_url.save(update_fields=["is_active"])
        state = "activated" if active else "deactivated"
        self.stdout.write(self.style.SUCCESS(f"{kb_url.title} {state}."))

    def _remove_url(self, url):
        from chatbot.models import KnowledgeBaseURL
        from chatbot.services import RAGService

        try:
            kb_url = KnowledgeBaseURL.objects.get(url=url)
        except KnowledgeBaseURL.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"URL not found: {url}"))
            return

        # Remove from FAISS
        import hashlib

        doc_id_prefix = f"web_{hashlib.md5(kb_url.url.encode()).hexdigest()[:12]}"
        try:
            rag_service = RAGService()
            rag_service.delete_document(doc_id_prefix)
            self.stdout.write("Removed chunks from vector store.")
        except Exception as e:
            self.stderr.write(f"Warning: could not remove from vector store: {e}")

        title = kb_url.title
        kb_url.delete()
        self.stdout.write(self.style.SUCCESS(f"Removed: {title}"))
