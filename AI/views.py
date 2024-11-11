from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings

import os
import pinecone
from pinecone import Pinecone, ServerlessSpec  # Import the Pinecone class

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Load environment variables from .env file
load_dotenv()
class AI(APIView):
    def post(self, request, *args, **kwargs):
        # Initialize the PDF loader
        loader = PyPDFLoader("C:\\Users\\yassi\\OneDrive\\Desktop\\FL\\AVL-Django\\AI\\testbook.pdf")
        pages = []

        # Use synchronous loading
        for page in loader.load():
            pages.append(page)

        # If no pages were loaded, return an error
        if not pages:
            return Response({"error": "Failed to load document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        print(f'You have {len(pages)} document(s) in your data')
        print(f'There are {len(pages[0].page_content)} characters in your sample document')
        print(f'Here is a sample: {pages[0].page_content[:200]}')

        # Split the text into chunks for vectorization
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        print(f'Now you have {len(texts)} documents after splitting')

        # Set up OpenAI embeddings
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        if not OPENAI_API_KEY:
            return Response({"error": "OpenAI API key is not set."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Initialize Pinecone client with your API key and environment
        pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        # Define your Pinecone index name
        index_name = "aitest"

        # Create the Pinecone index if it doesn't exist
        try:
            # Try to connect to the Pinecone index
            index = pinecone_client.Index(index_name)
        except Exception as e:
            # Optionally, create the index if it doesn't exist
            print(f"Index '{index_name}' does not exist. Creating it now...")
            # Assuming OpenAI's embedding dimension is 1536 (change if using a different model)
            pinecone_client.create_index(
                name=index_name,
                dimension=3072,  # You can change this if you're using a different model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'  # Specify your region
                )
            )
            index = pinecone_client.Index(index_name)

        # Prepare vectors and upsert them into Pinecone
        upsert_data = []
        for i, text in enumerate(texts):
            vector = embeddings.embed_documents([text.page_content])[0]  # Get the embedding for the document
            upsert_data.append((f"doc_{i}", vector, {"text": text.page_content}))
        # Upsert vectors into the Pinecone index
        index.upsert(vectors=upsert_data)
        print("Upsert data:", upsert_data)

        # Perform a similarity search query
        query = request.data.get("query", "What is great about having kids?")
        query_vector = embeddings.embed_query(query)  # Get the embedding for the query

        # Query Pinecone for similar documents
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        print('__________________________________________')
        print("Pinecone query results:", results)
        print('__________________________________________')
        # Prepare and return the results
        if 'matches' in results and results['matches']:
            response_data = [match['metadata']['text'] for match in results['matches'] if 'metadata' in match]
            return Response({"results": response_data}, status=status.HTTP_200_OK)
        else:
            print("No matches found in Pinecone results.")
            return Response({"error": "No matches found."}, status=status.HTTP_404_NOT_FOUND)
