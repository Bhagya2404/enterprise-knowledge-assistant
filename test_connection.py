import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

print("Loading Environment Variables...")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("API_VERSION")
)

print("Connected to Azure OpenAI")

response = client.chat.completions.create(
    model=os.getenv("GPT41_DEPLOYMENT"),
    messages=[
        {
            "role": "user",
            "content": "What is Generative AI?"
        }
    ]
)

print("\nModel Response:\n")
print(response.choices[0].message.content)