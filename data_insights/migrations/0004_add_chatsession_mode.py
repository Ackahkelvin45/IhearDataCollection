from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("data_insights", "0003_alter_chatsession_title"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="mode",
            field=models.CharField(
                choices=[("analysis", "Analysis"), ("ml", "Ml")],
                default="analysis",
                max_length=20,
            ),
        ),
    ]
