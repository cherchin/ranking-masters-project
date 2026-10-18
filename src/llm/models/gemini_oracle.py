
import os
import time
import dotenv
from google import genai
from google.genai import types

from prompt import ORACLE_SYSTEM, oracle_prompt

dotenv.load_dotenv()

class Gemini_Oracle:
    def __init__(self, model_name=None):
        api_key = os.environ.get("GEMINI_API_KEY")
        model_name = os.environ.get("GEMINI_MODEL")

        self.client = genai.Client(api_key=api_key)

        # Set GEMINI_MODEL to a model enabled for your API key.
        self.model_name = (
            model_name
        )

    def generate_gold_response(self, task, persona):
        prompt = oracle_prompt(
            task=task,
            persona=persona,
        )

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=ORACLE_SYSTEM.strip(),
                        temperature=0.2,
                        max_output_tokens=400,
                    ),
                )

                text = (response.text or "").strip()

                if not text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return text

            except Exception:
                if attempt == 2:
                    raise

                time.sleep(2 ** attempt)

        raise RuntimeError("Oracle generation failed.")
