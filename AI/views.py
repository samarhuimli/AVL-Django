from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangChainPinecone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from avl.settings import PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_ENVIRONMENT

# Load environment variables from .env file
load_dotenv()
embed_model = "text-embedding-ada-002"


class AI(APIView):
    def post(self, request, *args, **kwargs):
        loader = PyPDFLoader("C:\\Users\\yassi\\OneDrive\\Desktop\\FL\\AVL-Django\\AI\\RichDadPoorDad.pdf")
        file_content = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=0,
            length_function=len,
        )
        book_texts = text_splitter.split_documents(file_content)

        # Create an instance of Pinecone
        pc = Pinecone(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )

        index_name = 'aitest'

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Check if the index exists, if not, create it
        if index_name not in pc.list_indexes().names():
            print(f"Index {index_name} does not exist.")
            # Create index if needed (optional based on your use case)
            pc.create_index(
                name=index_name,
                dimension=1536,  # Adjust the dimension to match your embeddings
                metric='euclidean',  # You can choose another metric depending on your use case
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )

        # Create the document search using LangChain's Pinecone class
        book_docsearch = LangChainPinecone.from_texts([t.page_content for t in book_texts], embeddings,
                                                      index_name=index_name)

        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        query = "give me the Contents of the book the big chapters of the book"
        docs = book_docsearch.similarity_search(query)

        chain = load_qa_chain(llm, chain_type="stuff")
        result = chain.run(input_documents=docs, question=query)

        return Response(result, status=status.HTTP_200_OK)

    # def post(self, request, *args, **kwargs):
    #     file_path = "C:\\Users\\yassi\\OneDrive\\Desktop\\FL\\AVL-Django\\AI\\RichDadPoorDad.pdf"
    #
    #     # Calculate file hash for uniqueness
    #     def calculate_file_hash(file_path):
    #         hasher = hashlib.md5()
    #         with open(file_path, 'rb') as f:
    #             while chunk := f.read(8192):
    #                 hasher.update(chunk)
    #         return hasher.hexdigest()
    #
    #     file_hash = calculate_file_hash(file_path)
    #
    #     # Load the PDF
    #     loader = PyPDFLoader(file_path)
    #     pages = [page for page in loader.load()]
    #
    #     if not pages:
    #         return Response({"error": "Failed to load document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    #
    #     print(f'Loaded {len(pages)} pages from the document.')
    #
    #     # Split the text into manageable chunks
    #     max_chunk_size = 3500  # Approximate number of characters that would fit within token limits
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=50)
    #     texts = text_splitter.split_documents(pages)
    #     print(f'Split document into {len(texts)} chunks.')
    #
    #     # OpenAI embeddings setup
    #     OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    #     if not OPENAI_API_KEY:
    #         return Response({"error": "OpenAI API key is not set."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    #
    #     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    #
    #     # Initialize Pinecone
    #     pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    #     index_name = "aitest"
    #
    #     # Create the Pinecone index if it doesn't exist
    #     try:
    #         index = pinecone_client.Index(index_name)
    #     except Exception:
    #         print(f"Index '{index_name}' does not exist. Creating it now...")
    #         pinecone_client.create_index(
    #             name=index_name,
    #             dimension=1536,  # Match OpenAI embedding dimension
    #             metric="cosine",
    #             spec=ServerlessSpec(cloud='aws', region='us-east-1')
    #         )
    #         index = pinecone_client.Index(index_name)
    #
    #     # Check if the document is already indexed
    #     existing = index.query(vector=None, filter={"file_hash": file_hash}, top_k=1)
    #     if existing and 'matches' in existing and existing['matches']:
    #         print(f"Document '{file_path}' already indexed.")
    #         return Response({"message": "Document already indexed."}, status=status.HTTP_200_OK)
    #
    #     # Upsert vectors into Pinecone
    #     upsert_data = []
    #     for i, text in enumerate(texts):
    #         vector = embeddings.embed_documents([text.page_content])[0]  # Get the vector from OpenAI embeddings
    #         metadata = {"text": text.page_content, "file_hash": file_hash}  # Optional metadata
    #
    #         # Make sure the ID is unique and vector is properly provided
    #         upsert_data.append((f"{file_hash}_{i}", vector, metadata))
    #
    #     # Perform the upsert operation with the correctly formatted data
    #     try:
    #         index.upsert(vectors=upsert_data)
    #         print(f"Indexed {len(upsert_data)} chunks from the document.")
    #
    #         # Query operation (replacing `vector=None` with a proper query format)
    #         query_response = index.query(vector=upsert_data[0][1], top_k=1)  # Querying using the first vector
    #         print("Query response:", query_response)
    #
    #         return Response({"message": "Document indexed and queried successfully."}, status=status.HTTP_201_CREATED)
    #     except Exception as e:
    #         print(f"Error during Pinecone upsert or query: {e}")
    #         return Response({"error": f"Failed to upsert/query vectors: {str(e)}"},
    #                         status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # def get(self, request, *args, **kwargs):
    #     chapters = self.extract_chapters_from_local_file()
    #     return JsonResponse({"chapters": chapters})
    #
    # def extract_chapters_from_local_file(self):
    #     OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    #     llm = OpenAI(api_key=OPENAI_API_KEY)
    #
    #     # Define the path to the PDF file
    #     pdf_path = os.path.join(os.path.dirname(__file__), "RichDadPoorDad.pdf")
    #
    #     # Open the PDF using pdfplumber
    #     with pdfplumber.open(pdf_path) as pdf_document:
    #         text_chunks = []
    #         chunk_size = 10  # Process 10 pages at a time
    #         chapters = []
    #
    #         # Extract text in chunks
    #         for i in range(0, len(pdf_document.pages), chunk_size):
    #             chunk_text = ""
    #             for page_num in range(i, min(i + chunk_size, len(pdf_document.pages))):
    #                 page = pdf_document.pages[page_num]
    #                 chunk_text += page.extract_text() + "\n"
    #
    #             text_chunks.append(chunk_text)
    #
    #         # Use OpenAI to identify chapters
    #         for idx, chunk in enumerate(text_chunks):
    #             prompt = (
    #                 "Analyze the PDF to extract a structured Table of Contents. If a clear TOC is not present, identify the major sections "
    #                 "and their approximate starting page numbers. Please prioritize accuracy over completeness. "
    #                 f"\n\n{chunk[:4000]}"
    #             )
    #             response = llm(prompt=prompt)
    #
    #             # Parse response for chapter information (adjust parsing as needed)
    #             for line in response.split("\n"):
    #                 if "Page" in line or "Chapter" in line:
    #                     parts = line.split("-")
    #                     if len(parts) == 2:
    #                         title = parts[0].strip()
    #                         page_info = parts[1].strip()
    #                         try:
    #                             page_number = int(page_info.split()[1])
    #                             chapters.append({
    #                                 "title": title,
    #                                 "approx_start_page": idx * chunk_size + page_number
    #                             })
    #                         except (IndexError, ValueError):
    #                             continue
    #
    #     return chapters