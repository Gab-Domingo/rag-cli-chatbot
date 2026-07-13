import os

from dotenv import load_dotenv
from google import genai

from ingest import EMBEDDING_MODEL, get_vectorstore, ingest_knowledge
from rag_pipeline import CHAT_MODEL, RAGPipeline


def main():
    load_dotenv()

    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    vectorstore = get_vectorstore()

    try:
        ingest_knowledge(client, vectorstore)
    except Exception as e:
        print(e)
        print("Initializing PoBot with existing knowledge base...")


    pipeline = RAGPipeline(vectorstore)

    print("\nRAG CLI Chatbot Initialized.")
    print(f"Using: LangChain + {EMBEDDING_MODEL} + {CHAT_MODEL}")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue

            print(" Searching Context....")
            result = pipeline.answer(user_input)
            print(f"\nPoBot:\n{result['answer']}")
            if result['citations']:
                print("\nSources: ")
                for source in result["citations"]:
                    print(f"- {source}")
            print("-" * 80)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break


if __name__ == "__main__":
    main()
