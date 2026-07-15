from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load DOCX files

loader = DirectoryLoader(
    "Data",
    glob="*.docx",
    loader_cls=Docx2txtLoader
)

documents = loader.load()

print("Documents Loaded:", len(documents))

# Chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Total Chunks Created:", len(chunks))

# Display first few chunks

for i, chunk in enumerate(chunks[:5]):
    print("\n----------------------")
    print("Chunk", i+1)
    print(chunk.page_content[:300])