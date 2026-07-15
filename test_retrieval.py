import os
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load Embedding Model

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT")
)

# Load Existing FAISS Index

vectorstore = FAISS.load_local(
    "vectordb",
    embeddings,
    allow_dangerous_deserialization=True
)

# Test Question

query = "How many casual leaves are available?"

results = vectorstore.similarity_search(
    query,
    k=3
)

print("\nQUESTION:")
print(query)

print("\nTOP MATCHES:\n")

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("-" * 50)
    print(doc.page_content)