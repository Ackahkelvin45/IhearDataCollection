import random
import uuid
from datetime import datetime, timedelta, timezone

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from core.models import (
    Category,
    Class,
    Community,
    Microphone_Type,
    Region,
    SubClass,
    Time_Of_Day,
)
from data.models import AudioFeature, NoiseAnalysis, NoiseDataset

User = get_user_model()

REGIONS = [
    ("Greater Accra", "Capital region, high urban density"),
    ("Ashanti", "Second-largest region, major commercial hub"),
    ("Western", "Coastal and mining communities"),
    ("Eastern", "Mountainous region with mixed urban-rural"),
]

COMMUNITIES = {
    "Greater Accra": ["Accra Central", "Tema", "Madina", "Kasoa"],
    "Ashanti": ["Kumasi Central", "Obuasi", "Ejisu", "Mampong"],
    "Western": ["Takoradi", "Tarkwa", "Axim", "Prestea"],
    "Eastern": ["Koforidua", "Nkawkaw", "Asamankese", "Suhum"],
}

CATEGORIES = [
    ("Traffic", "Road traffic and vehicle noise"),
    ("Construction", "Building and infrastructure construction"),
    ("Market", "Outdoor market and commercial activity"),
    ("Industrial", "Factory and industrial machinery"),
    ("Community Events", "Social gatherings, ceremonies, and community activities"),
]

CLASSES = {
    "Traffic": ["Heavy Vehicles", "Light Vehicles", "Motorcycles"],
    "Construction": ["Heavy Machinery", "Hand Tools", "Drilling"],
    "Market": ["Vendor Activity", "Crowd Noise", "Public Address"],
    "Industrial": ["Machinery Hum", "Impact Noise", "Ventilation Systems"],
    "Community Events": ["Music/DJ", "Ceremony", "Sports"],
}

SUBCLASSES = {
    "Heavy Vehicles": ["Trucks", "Buses"],
    "Light Vehicles": ["Cars", "Taxis"],
    "Motorcycles": ["Motorbikes", "Tricycles"],
    "Heavy Machinery": ["Excavators", "Bulldozers"],
    "Hand Tools": ["Hammers", "Drills"],
    "Drilling": ["Pneumatic", "Rotary"],
    "Vendor Activity": ["Food Vendors", "Cloth Vendors"],
    "Crowd Noise": ["Dense Crowd", "Sparse Crowd"],
    "Public Address": ["Loudspeakers", "Megaphones"],
    "Machinery Hum": ["Generators", "Motors"],
    "Impact Noise": ["Metal Stamping", "Hammering"],
    "Ventilation Systems": ["HVAC", "Fans"],
    "Music/DJ": ["Amplified Music", "Live Band"],
    "Ceremony": ["Funerals", "Weddings"],
    "Sports": ["Football", "Athletics"],
}

MICROPHONE_TYPES = [
    ("Condenser", "Studio-grade condenser microphone"),
    ("Dynamic", "Handheld dynamic microphone"),
    ("Lavalier", "Clip-on lapel microphone"),
]

TIMES_OF_DAY = [
    ("Morning", "6am – 12pm"),
    ("Afternoon", "12pm – 6pm"),
    ("Night", "6pm – 6am"),
]

RECORDING_DEVICES = [
    "iPhone 15 Pro",
    "Samsung Galaxy S24",
    "Zoom H4n",
    "Tascam DR-40X",
    "Roland R-07",
]

# Typical dB ranges per category (mean, spread)
DB_PROFILES = {
    "Traffic": (72, 12),
    "Construction": (85, 10),
    "Market": (68, 8),
    "Industrial": (90, 8),
    "Community Events": (78, 14),
}


class Command(BaseCommand):
    help = "Seed the database with test noise dataset records for development"

    def add_arguments(self, parser):
        parser.add_argument(
            "--count",
            type=int,
            default=50,
            help="Number of NoiseDataset records to create (default: 50)",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete all existing seed data before creating new records",
        )

    def handle(self, *args, **options):
        if options["clear"]:
            deleted, _ = NoiseDataset.objects.filter(
                description__startswith="[SEED]"
            ).delete()
            self.stdout.write(self.style.WARNING(f"Cleared {deleted} seeded records"))

        self.stdout.write("Creating lookup data...")
        regions = self._create_regions()
        categories = self._create_categories()
        mic_types = self._create_mic_types()
        times_of_day = self._create_times_of_day()
        collector = self._get_or_create_collector()

        count = options["count"]
        self.stdout.write(f"Creating {count} noise dataset records...")
        created = 0

        for i in range(count):
            region_name, region = random.choice(list(regions.items()))
            community = random.choice(list(Community.objects.filter(region=region)))
            category_name, category = random.choice(list(categories.items()))
            class_obj = random.choice(list(Class.objects.filter(category=category)))
            subclass = random.choice(
                list(SubClass.objects.filter(parent_class=class_obj))
            )

            noise_id = f"SEED-{uuid.uuid4().hex[:8].upper()}"
            days_ago = random.randint(0, 365)
            rec_date = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)

            dataset = NoiseDataset(
                name=f"{category_name} – {region_name} #{i + 1}",
                collector=collector,
                description=f"[SEED] Test record {i + 1}: {category_name} noise in {community.name}",
                region=region,
                category=category,
                time_of_day=random.choice(list(times_of_day.values())),
                community=community,
                class_name=class_obj,
                subclass=subclass,
                microphone_type=random.choice(list(mic_types.values())),
                audio="files/seed_placeholder.wav",
                recording_date=rec_date,
                recording_device=random.choice(RECORDING_DEVICES),
                noise_id=noise_id,
            )
            dataset.save()

            self._create_noise_analysis(dataset, category_name)
            self._create_audio_features(dataset)
            created += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Done. Created {created} NoiseDataset records with linked "
                "NoiseAnalysis and AudioFeature entries."
            )
        )
        self.stdout.write(
            "Run with --clear to wipe and re-seed, or --count N to change volume."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_regions(self):
        result = {}
        for region_name, desc in REGIONS:
            region, _ = Region.objects.get_or_create(
                name=region_name, defaults={"description": desc}
            )
            result[region_name] = region
            for community_name in COMMUNITIES[region_name]:
                Community.objects.get_or_create(name=community_name, region=region)
        return result

    def _create_categories(self):
        from data.models import Dataset as DatasetType

        noise_dataset_type, _ = DatasetType.objects.get_or_create(
            name="noise", defaults={"description": "Noise dataset"}
        )
        result = {}
        for cat_name, desc in CATEGORIES:
            category, _ = Category.objects.get_or_create(
                name=cat_name,
                defaults={"description": desc, "data_type": noise_dataset_type},
            )
            result[cat_name] = category
            for class_name in CLASSES[cat_name]:
                class_obj, _ = Class.objects.get_or_create(
                    name=class_name, category=category
                )
                for subclass_name in SUBCLASSES.get(class_name, []):
                    SubClass.objects.get_or_create(
                        name=subclass_name, parent_class=class_obj
                    )
        return result

    def _create_mic_types(self):
        result = {}
        for name, desc in MICROPHONE_TYPES:
            mic, _ = Microphone_Type.objects.get_or_create(
                name=name, defaults={"description": desc}
            )
            result[name] = mic
        return result

    def _create_times_of_day(self):
        result = {}
        for name, desc in TIMES_OF_DAY:
            tod, _ = Time_Of_Day.objects.get_or_create(
                name=name, defaults={"description": desc}
            )
            result[name] = tod
        return result

    def _get_or_create_collector(self):
        user, created = User.objects.get_or_create(
            username="seed_collector",
            defaults={
                "email": "seed@example.com",
                "first_name": "Seed",
                "last_name": "Collector",
            },
        )
        if created:
            user.set_password("seedpassword123")
            user.save()
            self.stdout.write("  Created seed_collector user")
        return user

    def _create_noise_analysis(self, dataset, category_name):
        mean_base, spread = DB_PROFILES.get(category_name, (70, 10))
        mean_db = round(random.gauss(mean_base, spread / 2), 2)
        noise_range = random.uniform(8, 25)
        max_db = round(mean_db + noise_range / 2, 2)
        min_db = round(mean_db - noise_range / 2, 2)
        std_db = round(random.uniform(2, noise_range / 3), 2)

        NoiseAnalysis.objects.create(
            noise_dataset=dataset,
            mean_db=mean_db,
            max_db=max_db,
            min_db=min_db,
            std_db=std_db,
            peak_count=random.randint(5, 80),
            peak_interval_mean=round(random.uniform(0.5, 10.0), 3),
            dominant_frequency=round(random.uniform(80, 4000), 1),
            frequency_range=f"{random.randint(20, 200)}-{random.randint(3000, 8000)}",
            event_count=random.randint(1, 20),
        )

    def _create_audio_features(self, dataset):
        AudioFeature.objects.create(
            noise_dataset=dataset,
            sample_rate=random.choice([16000, 22050, 44100, 48000]),
            num_samples=random.randint(44100, 441000),
            duration=round(random.uniform(5.0, 120.0), 2),
            rms_energy=round(random.uniform(0.01, 0.4), 4),
            zero_crossing_rate=round(random.uniform(0.02, 0.3), 4),
            spectral_centroid=round(random.uniform(500, 4000), 2),
            spectral_bandwidth=round(random.uniform(200, 3000), 2),
            spectral_rolloff=round(random.uniform(1000, 8000), 2),
            spectral_flatness=round(random.uniform(0.001, 0.5), 4),
            harmonic_ratio=round(random.uniform(0.1, 0.9), 3),
            percussive_ratio=round(random.uniform(0.1, 0.9), 3),
        )
