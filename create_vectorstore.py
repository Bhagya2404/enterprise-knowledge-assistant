import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load documents

loader = DirectoryLoader(
    "Data",
    glob="*.docx",
    loader_cls=Docx2txtLoader
)

documents = loader.load()

print(f"Documents Loaded: {len(documents)}")

# Split into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Chunks Created: {len(chunks)}")

# Azure OpenAI Embeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT")
)

print("Generating embeddings...")

# Create FAISS vector store

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Save locally

vectorstore.save_local("vectordb")

print("✅ FAISS Vector Store Created Successfully")