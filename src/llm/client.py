import os
import warnings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from prompt import gen_active_learning, build_context, gen_questions, identify_preference_type
# Filter non-critical SSL and deprecation warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
if not gemini_model:
    raise ValueError("GEMINI_MODEL not found in .env file.")

def main():
    # domain and candidate items hardcoded for now
    domain = "stocks"

    candidate_items = """
    AAPL - Apple Inc.
    TSLA - Tesla Inc.
    NVDA - NVIDIA Corporation
    MSFT - Microsoft Corporation
    AMZN - Amazon.com Inc.
    """
    elicitation_transcript = ""

    # Build domain-specific context
    context = build_context(
        domain=domain,
        candidate_items=candidate_items,
        elicitation_transcript=elicitation_transcript
    )

    # Combine generic active learning instructions
    # with domain-specific ranking context
    system_instruction = f"""
        {gen_active_learning(domain, candidate_items, elicitation_transcript)}
        {gen_questions(domain, candidate_items, elicitation_transcript)}
        {identify_preference_type(domain, candidate_items, elicitation_transcript)}
        {context}
    """
    # Create Gemini client
    client = genai.Client(api_key=gemini_api_key)

    chat = client.chats.create(
        model=gemini_model,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    
    print("Start chatting. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            response = chat.send_message(user_input)
            print(f"Model: {response.text}")
        except Exception as error:
            print(f"Model request failed: {error}")


if __name__ == "__main__":
    main()