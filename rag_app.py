import os
from dotenv import load_dotenv

from openai import AzureOpenAI

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Azure OpenAI Client

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("API_VERSION")
)

# Embedding Model

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT")
)

# Load FAISS

vectorstore = FAISS.load_local(
    "vectordb",
    embeddings,
    allow_dangerous_deserialization=True
)

question = input("Ask your question: ")

# Retrieve documents

docs = vectorstore.similarity_search(
    question,
    k=3
)

context = "\n\n".join(
    [doc.page_content for doc in docs]
)

prompt = f"""
You are an Enterprise Knowledge Assistant.

Use ONLY the information contained in the context.

If the answer is not available in the context,
respond with:

"I could not find that information in the company documents."

Context:
{context}

Question:
{question}

Answer:
"""

response = client.chat.completions.create(
    model=os.getenv("GPT41_DEPLOYMENT"),
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n----- ANSWER -----\n")
print(response.choices[0].message.content)