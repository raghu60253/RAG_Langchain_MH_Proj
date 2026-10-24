# from openai import OpenAI
# client = OpenAI(
#     base_url="http://127.0.0.1:8888/v1",
#     api_key="YOUR_API_KEY"
# )
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello! Can you help me with a question?"}],
#         stream=True,
# )
# for chunk in response:
#     print(chunk.choices[0].delta, end="", flush=True)
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()
llm = ChatOllama(
    model=os.getenv("OLLAMA_CHAT_MODEL", "kimi-k2.6:cloud"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

for chunk in llm.stream(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Hello! Can you help me with a question?"),
    ]
):
    print(chunk.content, end="", flush=True)