# document_upload/models.py
from django.db import models

class Document(models.Model):
    file = models.FileField(upload_to='uploads/')  # Store files in 'uploads/' directory inside MEDIA_ROOT
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name  # Display the file name in the admin
class Book(models.Model):
    user = models.CharField(max_length=255)
    url = models.URLField()
    name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name