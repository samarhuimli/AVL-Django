# serializers.py
from rest_framework import serializers
from .models import Document  # Importation du modèle Document

# Définition d'un sérialiseur pour le modèle Document
class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document  # Spécifie le modèle à sérialiser
        fields = ['file', 'uploaded_at']  # Liste des champs à inclure dans la sérialisation
