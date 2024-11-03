
# document_upload/urls.py
from django.urls import path
from .views import DocumentUploadView  # Assurez-vous d'importer la nouvelle vue

urlpatterns = [
    path('api/upload/', DocumentUploadView.as_view(), name='file-upload'),  # URL pour le téléchargement de fichiers
]

