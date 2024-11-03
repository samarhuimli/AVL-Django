# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Document
from .serializers import DocumentSerializer

class DocumentUploadView(APIView):
    def post(self, request, *args, **kwargs):
        # Crée une instance du sérialiseur avec les données de la requête
        serializer = DocumentSerializer(data=request.data)
        if serializer.is_valid():  # Valide les données
            serializer.save()  # Enregistre le fichier dans la base de données
            return Response(serializer.data, status=status.HTTP_201_CREATED)  # Répond avec les données du sérialiseur
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)  # Répond avec les erreurs de validation

    def get(self, request, *args, **kwargs):
        # Récupère tous les documents existants (facultatif)
        documents = Document.objects.all()
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data)  # Répond avec la liste des documents
