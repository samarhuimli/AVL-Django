from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Document
from .serializers import DocumentSerializer

from dotenv import load_dotenv
import os  # For handling file names
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, BotoCoreError
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from pinecone import Pinecone, ServerlessSpec
from avl.settings import PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_ENVIRONMENT

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME", "us-east-1")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")


class DocumentUploadView(APIView):
    def post(self, request, *args, **kwargs):
        # AWS S3 setup
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_S3_REGION_NAME,
        )

        # Get the uploaded file
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)
        file.name = os.path.splitext(file.name)[0].lower().replace(' ', '_')

        try:
            # Upload the file to AWS S3
            s3.upload_fileobj(
                file,
                AWS_STORAGE_BUCKET_NAME,
                file.name,  # Use the file name as the S3 key
                ExtraArgs={'ContentType': file.content_type}
            )

            # Generate the file URL
            file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{file.name}"
            print(f"------------------------------------------ {file.name}")  # Debugging log

            if not file.name:
                return Response({"error": "File name is missing or invalid."}, status=status.HTTP_400_BAD_REQUEST)

            # Derive index name from the file name (remove extension)
            index_name = os.path.splitext(file.name)[0].lower().replace(' ', '_')

            # Indexing the file
            loader = PyPDFLoader(file_url)  # Replace local file path with the URL
            file_content = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=0,
                length_function=len,
            )
            book_texts = text_splitter.split_documents(file_content)

            # Create Pinecone index
            pc = Pinecone(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENVIRONMENT
            )

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            print(f"Uploaded file name:;;;;;;;;;;;; {index_name}")

            if index_name not in pc.list_indexes().names():
                print(f"Creating new Pinecone index: {index_name}")  # Debugging log
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric='euclidean',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            else:
                print(f"Using existing Pinecone index: {index_name}")  # Debugging log
            print(f"------------------------------------------ {index_name}")  # Debugging log

            book_docsearch = LangChainPinecone.from_texts(
                [t.page_content for t in book_texts], embeddings, index_name=index_name
            )
            print(f"------------------------------------------ {index_name}")  # Debugging log

            llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            query = "give me a recap of the ninth chapter of the book"
            docs = book_docsearch.similarity_search(query)

            chain = load_qa_chain(llm, chain_type="stuff")
            result = chain.run(input_documents=docs, question=query)
            print(f"Using existing Pinecone index bf response: {index_name}")  # Debugging log

            return Response({
                "message": "File processed successfully!",
                "result": result,
                "url": file_url,
                "index_name": index_name
            }, status=status.HTTP_200_OK)

        except (NoCredentialsError, PartialCredentialsError):
            return Response({"error": "Invalid AWS credentials. Check your keys."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except BotoCoreError as e:
            return Response({"error": "AWS S3 error occurred.", "details": str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({"error": "An unexpected error occurred.", "details": str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)



    # def post(self, request):
    #     """
    #     Gère le téléchargement d'un fichier vers AWS S3 et enregistre ses métadonnées dans la base de données.
    #     """
    #     # Vérifiez si les variables AWS sont correctement configurées
    #     if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not AWS_STORAGE_BUCKET_NAME:
    #         return JsonResponse(
    #             {"error": "AWS configuration missing. Check environment variables."},
    #             status=500
    #         )
    #
    #     # Récupérez le fichier téléchargé
    #     file = request.FILES.get('file')
    #     if not file:
    #         return JsonResponse({"error": "No file uploaded."}, status=400)
    #
    #     # Valider la taille et le type du fichier
    #     # if file.size > 10 * 1024 * 1024:  # Limite de taille : 10 Mo
    #     #     return JsonResponse({"error": "File size exceeds 10MB limit."}, status=400)
    #
    #     # if file.content_type not in ["image/jpeg", "image/png", "application/pdf", "text/plain"]:
    #     #     return JsonResponse({"error": f"Unsupported file type: {file.content_type}"}, status=400)
    #
    #     # Initialiser le client S3
    #     s3 = boto3.client(
    #         's3',
    #         aws_access_key_id=AWS_ACCESS_KEY_ID,
    #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    #         region_name=AWS_S3_REGION_NAME,
    #     )
    #
    #     try:
    #         # Télécharger le fichier vers S3
    #         s3.upload_fileobj(
    #             file,
    #             AWS_STORAGE_BUCKET_NAME,
    #             file.name,  # Utiliser le nom du fichier comme clé dans S3
    #             ExtraArgs={'ContentType': file.content_type}
    #         )
    #
    #         # Générer l'URL du fichier
    #         file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{file.name}"
    #
    #         # Enregistrer les métadonnées dans la base de données
    #         serializer = DocumentSerializer(data={"name": file.name, "url": file_url})
    #         if serializer.is_valid():
    #             serializer.save()
    #         else:
    #             return JsonResponse({"error": "Failed to save document metadata.", "details": serializer.errors}, status=500)
    #
    #         return JsonResponse({"message": "File uploaded successfully!", "url": file_url})
    #
    #     except (NoCredentialsError, PartialCredentialsError):
    #         return JsonResponse({"error": "Invalid AWS credentials. Check your keys."}, status=500)
    #     except BotoCoreError as e:
    #         return JsonResponse({"error": "AWS S3 error occurred.", "details": str(e)}, status=500)
    #     except Exception as e:
    #         return JsonResponse({"error": "An unexpected error occurred.", "details": str(e)}, status=500)

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
