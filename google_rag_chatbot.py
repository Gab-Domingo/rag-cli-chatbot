import os

from dotenv import load_dotenv
from google import genai

from ingest import get_collection, ingest_knowledge
from rag_pipeline import CHAT_MODEL, EMBEDDING_MODEL, RAGPipeline


def main():
    load_dotenv()

    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    collection = get_collection()

    ingest_knowledge(client, collection)

    pipeline = RAGPipeline(client, collection)

    print("\nRAG CLI Chatbot Initialized.")
    print(f"Using: {EMBEDDING_MODEL} + {CHAT_MODEL}")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue

            print(" Searching Context....")
            answer = pipeline.answer(user_input)
            print(f"\nPoBot: {answer}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break


if __name__ == "__main__":
    main()
