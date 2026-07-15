import os
import re

from dotenv import load_dotenv

from openai import AzureOpenAI

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Memory

memory = {}

# Sample Tickets

tickets = {
    "INC1001": "VPN Issue",
    "INC1002": "Password Reset",
    "INC1003": "Laptop Replacement"
}

# Approval Keywords

sensitive_actions = [
    "delete",
    "terminate",
    "reset account",
    "access escalation",
    "disable user"
]

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("API_VERSION")
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT")
)

vectorstore = FAISS.load_local(
    "vectordb",
    embeddings,
    allow_dangerous_deserialization=True
)

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    # Approval Check

    if any(
        action in question.lower()
        for action in sensitive_actions
    ):
        print("\n⚠ APPROVAL REQUIRED")
        continue

    # Calculator Tool

    if "calculate" in question.lower():

        expression = (
            question.lower()
            .replace("calculate", "")
            .strip()
        )

        try:
            result = eval(expression)
            print("\nResult:", result)

        except:
            print("\nInvalid Expression")

        continue

    # Ticket Tool

    ticket = re.findall(
        r"INC\d+",
        question.upper()
    )

    if ticket:

        print(
            "\nTicket:",
            tickets.get(
                ticket[0],
                "Ticket Not Found"
            )
        )

        continue

    # RAG

    docs = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Use ONLY the provided context.

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model=os.getenv("GPT41_DEPLOYMENT"),
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    print(
        "\nAnswer:\n",
        response.choices[0].message.content
    )