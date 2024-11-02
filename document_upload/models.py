# document_upload/models.py
from django.db import models

class Document(models.Model):
    file = models.FileField(upload_to='uploads/')  # Chemin où le fichier sera enregistré
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name  # Afficher le nom du fichier dans l'admin
