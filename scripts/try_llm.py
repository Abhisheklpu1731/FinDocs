import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()   # reads .env into environment variables

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",   # send requests to Groq, not OpenAI
)

response = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[{"role": "user", "content": "In one sentence, what is an annual report?"}],
)

print(response.choices[0].message.content)
print("Tokens used:", response.usage.total_tokens)