import os
import warnings
from prompt import gen_active_learning
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types


# Filter non-critical SSL and deprecation warnings
warnings.filterwarnings("ignore")

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
if not gemini_model:
    raise ValueError("GEMINI_MODEL not found in .env file.")

client = genai.Client(api_key=gemini_api_key)
print(f"Loaded MODEL={os.getenv('GEMINI_MODEL')}, using Gemini client")

previous_queries = ["AAPL", "GOOGL", "AMZN"]
user_input = f"Previously queried edge cases: {previous_queries}."

response = client.models.generate_content(
    model=gemini_model,
    contents=user_input,
    config=types.GenerateContentConfig(
        system_instruction=gen_active_learning,
    )
)

print("\n --- Model Output ---")
print(response.text)