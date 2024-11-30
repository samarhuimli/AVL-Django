
# document_upload/urls.py
from django.urls import path

from .booksview import UserBooksView
from .views import DocumentUploadView  # Assurez-vous d'importer la nouvelle vue

urlpatterns = [
    path('api/upload/', DocumentUploadView.as_view(), name='file-upload'),  # URL pour le téléchargement de fichiers
    path('api/books/', UserBooksView.as_view(), name='user-books'),  # URL pour obtenir les livres de l'utilisateur
]

