from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("data_insights", "0004_add_chatsession_mode"),
    ]

    operations = [
        migrations.RenameField(
            model_name="chatmessage",
            old_name="visulization",
            new_name="visualization",
        ),
    ]
