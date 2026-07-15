import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("API_VERSION")
)

PROMPT = """
Explain the leave approval workflow in the company.
"""

models = [
    os.getenv("GPT41_DEPLOYMENT"),
    os.getenv("GPT51_DEPLOYMENT")
]

for model in models:

    print("\n" + "=" * 60)
    print("MODEL:", model)
    print("=" * 60)

    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT
            }
        ]
    )

    end_time = time.time()

    answer = response.choices[0].message.content

    print("\nResponse:\n")
    print(answer)

    print("\nResponse Time:",
          round(end_time - start_time, 2),
          "seconds")