# Generated by Django 5.0.4 on 2024-04-09 10:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_pdf_delete_chat'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pdf',
            name='file',
            field=models.FileField(upload_to='files'),
        ),
    ]
