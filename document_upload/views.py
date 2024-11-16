from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Document
from .serializers import DocumentSerializer
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, BotoCoreError
from django.http import JsonResponse
from django.views import View
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Configuration AWS à partir des variables d'environnement
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME", "us-east-1")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")


class DocumentUploadView(APIView):
    def post(self, request):
        """
        Gère le téléchargement d'un fichier vers AWS S3 et enregistre ses métadonnées dans la base de données.
        """
        # Vérifiez si les variables AWS sont correctement configurées
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not AWS_STORAGE_BUCKET_NAME:
            return JsonResponse(
                {"error": "AWS configuration missing. Check environment variables."},
                status=500
            )

        # Récupérez le fichier téléchargé
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({"error": "No file uploaded."}, status=400)

        # Valider la taille et le type du fichier
        # if file.size > 10 * 1024 * 1024:  # Limite de taille : 10 Mo
        #     return JsonResponse({"error": "File size exceeds 10MB limit."}, status=400)

        # if file.content_type not in ["image/jpeg", "image/png", "application/pdf", "text/plain"]:
        #     return JsonResponse({"error": f"Unsupported file type: {file.content_type}"}, status=400)

        # Initialiser le client S3
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_S3_REGION_NAME,
        )

        try:
            # Télécharger le fichier vers S3
            s3.upload_fileobj(
                file,
                AWS_STORAGE_BUCKET_NAME,
                file.name,  # Utiliser le nom du fichier comme clé dans S3
                ExtraArgs={'ContentType': file.content_type}
            )

            # Générer l'URL du fichier
            file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{file.name}"

            # Enregistrer les métadonnées dans la base de données
            serializer = DocumentSerializer(data={"name": file.name, "url": file_url})
            if serializer.is_valid():
                serializer.save()
            else:
                return JsonResponse({"error": "Failed to save document metadata.", "details": serializer.errors}, status=500)

            return JsonResponse({"message": "File uploaded successfully!", "url": file_url})

        except (NoCredentialsError, PartialCredentialsError):
            return JsonResponse({"error": "Invalid AWS credentials. Check your keys."}, status=500)
        except BotoCoreError as e:
            return JsonResponse({"error": "AWS S3 error occurred.", "details": str(e)}, status=500)
        except Exception as e:
            return JsonResponse({"error": "An unexpected error occurred.", "details": str(e)}, status=500)

    def get(self, request):
        """
        Récupère tous les documents enregistrés dans la base de données.
        """
        documents = Document.objects.all()
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    # def post(self, request, *args, **kwargs):
    #     # Crée une instance du sérialiseur avec les données de la requête
    #     serializer = DocumentSerializer(data=request.data)
    #     if serializer.is_valid():  # Valide les données
    #         serializer.save()  # Enregistre le fichier dans la base de données
    #         return Response(serializer.data, status=status.HTTP_201_CREATED)  # Répond avec les données du sérialiseur
    #     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)  # Répond avec les erreurs de validation

    def get(self, request, *args, **kwargs):
        # Récupère tous les documents existants (facultatif)
        documents = Document.objects.all()
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data)  # Répond avec la liste des documents
