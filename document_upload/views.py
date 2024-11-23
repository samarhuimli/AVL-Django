from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
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
from avl.settings import PINECONE_API_KEY, OPENAI_API_KEY, PINECONE_ENVIRONMENT , TAVILY_API_KEY
from tavily import TavilyClient, MissingAPIKeyError
from langchain_community.tools import TavilySearchResults
from functools import wraps

def chain(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Starting chain...")
        result = func(*args, **kwargs)
        print("Ending chain...")
        return result
    return wrapper

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME", "us-east-1")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")

tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

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

            return Response({
                "message": "File processed successfully!",
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

    @chain
    def tool_chain(user_input: str, config: dict):
        # Step 1: Prepare the input
        input_ = {"user_input": user_input}

        # Step 2: Invoke the LLM chain
        print("Invoking LLM chain...")
        ai_msg = llm_chain.invoke(input_, config=config)

        # Step 3: Process tool calls
        print("Processing tool calls...")
        tool_msgs = tool.batch(ai_msg.tool_calls, config=config)

        # Step 4: Return the final output
        return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)

    def get(self, request, *args, **kwargs):

        try:
            # Extract the user_id and book_name from query params or headers
            user_id = request.query_params.get('user_id')
            book_name = request.query_params.get('book_name')
            if not user_id or not book_name:
                return Response({"error": "User ID and Book Name are required."}, status=status.HTTP_400_BAD_REQUEST)
            # Construct index name based on user_id and book_name
            index_name = book_name
            print(f"------------------------------------------ {index_name}")  # Debugging log
            if not index_name:
                return Response({"error": "Index not found."}, status=status.HTTP_400_BAD_REQUEST)

            # Initialize Pinecone and OpenAI
            pinecone_client = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

            # Connect to the Pinecone index
            index = pinecone_client.Index(index_name)

            # Initialize LangChain Pinecone vector store
            book_docsearch = LangChainPinecone(index, embeddings.embed_query, text_key='text')

            # Check action type
            action_type = request.query_params.get('action_type')
            if action_type == 'resume':
                # Get specific chapter or entire book
                chapter = request.query_params.get('chapter')
                llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

                if chapter:
                    query = f"Summarize chapter {chapter} of the book."
                else:
                    query = "Summarize the entire book."

                docs = book_docsearch.similarity_search(query)
                chain = load_qa_chain(llm, chain_type="stuff")
                result = chain.run(input_documents=docs, question=query)

                return Response({"summary": result}, status=status.HTTP_200_OK)

            elif action_type == 'DeepDive':

                try:

                    ddq_query = request.query_params.get('DDQ')

                    if not ddq_query:
                        return Response({"error": "The DDQ query parameter is required."},

                                        status=status.HTTP_400_BAD_REQUEST)

                    # Initialize TavilySearchResults tool

                    tavily_tool = TavilySearchResults(

                        max_results=5,  # Limit to top 5 results

                        search_depth="advanced",

                        include_answer=True,  # Not summarizing, just fetching results

                        include_raw_content=True,

                        include_images=True,


                    )

                    search_results = tavily_tool.run(ddq_query)
                    print(search_results)
                    results = []

                    for result in search_results:
                        # Extract content preview (first few lines) and link

                        preview = result.get("content", "").split("\n")[0:3]  # First few lines

                        link = result.get("url", "No link available")

                        # Append the result

                        results.append({

                            "preview": " ".join(preview),

                            "link": link,

                        })

                    # Return the results as a response

                    return Response({"results": results}, status=status.HTTP_200_OK)

                except Exception as e:

                    return Response({"error": "An error occurred during DeepDive processing.", "details": str(e)},

                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            elif action_type == 'question':
                try:
                    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
                except MissingAPIKeyError:
                    print("API key is missing. Please provide a valid API key.")
                # Answer a specific question
                query = request.query_params.get('query')
                if not query:
                    return Response({"error": "Query parameter is required for question action."},
                                    status=status.HTTP_400_BAD_REQUEST)

                llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
                docs = book_docsearch.similarity_search(query)
                chain = load_qa_chain(llm, chain_type="stuff")
                result = chain.run(input_documents=docs, question=query)

                return Response({"answer": result}, status=status.HTTP_200_OK)

            else:
                return Response({"error": "Invalid action_type. Use 'resume' or 'question'."},
                                status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({"error": "An unexpected error occurred.", "details": str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)


