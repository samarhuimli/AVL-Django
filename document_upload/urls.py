# document_upload/urls.py
from django.urls import path
from .views import search_and_upload

urlpatterns = [
    path('search-upload/', search_and_upload, name='search_upload'),
]
