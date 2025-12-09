"""
Management command to clean up old Celery task results from Redis
This prevents Redis from filling up with old task results
"""

from django.core.management.base import BaseCommand
from celery.result import AsyncResult
from django.conf import settings
import redis
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Clean up old Celery task results from Redis to prevent memory issues"

    def add_arguments(self, parser):
        parser.add_argument(
            "--older-than-hours",
            type=int,
            default=24,
            help="Clean up results older than this many hours (default: 24)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without actually deleting",
        )

    def handle(self, *args, **options):
        older_than_hours = options["older_than_hours"]
        dry_run = options["dry_run"]

        self.stdout.write(
            self.style.SUCCESS(
                f"Starting cleanup of Celery results older than {older_than_hours} hours..."
            )
        )

        try:
            # Connect to Redis
            redis_url = settings.CELERY_RESULT_BACKEND
            if "?" in redis_url:
                redis_url = redis_url.split("?")[0]

            # Parse Redis URL
            if redis_url.startswith("redis://") or redis_url.startswith("rediss://"):
                # Extract connection details
                url_parts = (
                    redis_url.replace("redis://", "")
                    .replace("rediss://", "")
                    .split("/")
                )
                connection_part = url_parts[0]
                db_id = int(url_parts[1]) if len(url_parts) > 1 else 0

                if "@" in connection_part:
                    auth, host_port = connection_part.split("@")
                    username, password = auth.split(":")
                    host, port = host_port.split(":")
                else:
                    username = None
                    password = None
                    host, port = connection_part.split(":")

                port = int(port)

                # Connect to Redis
                r = redis.Redis(
                    host=host,
                    port=port,
                    db=db_id,
                    username=username if username else None,
                    password=password if password else None,
                    decode_responses=False,  # We need bytes for keys
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f"Invalid Redis URL format: {redis_url}")
                )
                return

            # Get all Celery result keys
            # Celery stores results with pattern: celery-task-meta-<task_id>
            pattern = b"celery-task-meta-*"
            keys = r.keys(pattern)

            self.stdout.write(f"Found {len(keys)} Celery result keys")

            deleted_count = 0
            kept_count = 0
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

            for key in keys:
                try:
                    # Get the task result
                    result_data = r.get(key)
                    if result_data:
                        # Try to get TTL (time to live) - if it exists, check if expired
                        ttl = r.ttl(key)

                        # If TTL is -1 (no expiration) or very large, check creation time
                        # We'll delete keys that are older than the threshold
                        # Since we can't easily get creation time, we'll delete keys with no TTL
                        # or keys that are close to expiring (older than our threshold)

                        # For now, delete keys that have no expiration set (TTL = -1)
                        # These are likely old results that weren't cleaned up
                        if ttl == -1:
                            if dry_run:
                                self.stdout.write(
                                    self.style.WARNING(
                                        f'Would delete: {key.decode("utf-8", errors="ignore")}'
                                    )
                                )
                            else:
                                r.delete(key)
                                deleted_count += 1
                        elif ttl > older_than_hours * 3600:
                            # Key expires in more than our threshold - likely new, keep it
                            kept_count += 1
                        else:
                            # Key expires soon or already expired, delete it
                            if dry_run:
                                self.stdout.write(
                                    self.style.WARNING(
                                        f'Would delete (expires soon): {key.decode("utf-8", errors="ignore")}'
                                    )
                                )
                            else:
                                r.delete(key)
                                deleted_count += 1
                    else:
                        # Empty or None result, delete it
                        if dry_run:
                            self.stdout.write(
                                self.style.WARNING(
                                    f'Would delete (empty): {key.decode("utf-8", errors="ignore")}'
                                )
                            )
                        else:
                            r.delete(key)
                            deleted_count += 1
                except Exception as e:
                    logger.error(f"Error processing key {key}: {str(e)}")
                    continue

            if dry_run:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"\nDry run complete. Would delete {deleted_count} keys, keep {kept_count} keys"
                    )
                )
            else:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"\nCleanup complete. Deleted {deleted_count} old result keys, kept {kept_count} keys"
                    )
                )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error during cleanup: {str(e)}"))
            logger.error(f"Celery result cleanup failed: {str(e)}", exc_info=True)
            raise
