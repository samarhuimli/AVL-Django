# document_upload/models.py
from django.db import models

class Document(models.Model):
    file = models.FileField(upload_to='uploads/')  # Store files in 'uploads/' directory inside MEDIA_ROOT
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name  # Display the file name in the admin
