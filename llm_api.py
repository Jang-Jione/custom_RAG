# llm_api.py
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # ✅ .env 파일 자동 로드

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def query_gpt(prompt: str, model: str = "gpt-4o-mini") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    return resp.choices[0].message.content
