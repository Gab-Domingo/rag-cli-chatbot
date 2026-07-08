import os
import glob
from google import genai
from google.genai import types
import chromadb

from dotenv import load_dotenv
load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY is not set")


#1. Initialize Google GenAI 
api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.create_collection(name='gooogle_rag_knowledge')

#2. Ingest Data, Generate Embeddings and Populate DB

def chunk_text(text, max_chars=1000, overlap=100):
    """Simple internal character chunker"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += (max_chars - overlap)
    return chunks

def ingest_knowledge():
    print("Scanning './knowledge' directory for files...")
    if not os.path.exists('./knowledge'):
        os.makedirs('./knowledge')
        print("Created empty './knowledge' folder. Please Drop PDFs or Images inside and re run")
    
    # Supported document and image types
    supported_extensions = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.txt"]
    all_files = []
    for ext in supported_extensions:
        all_files.extend(glob.glob(os.path.join("./knowledge",ext)))

    if not all_files:
        print("No valid documents found in './knowledge.Add files to begin.")
        return

    all_chunks = []
    all_ids = []
    chunk_counter = 0

    for file_path in all_files:
        filename = os.path.basename(file_path)
        ext = filename.split(".")[-1].lower()

        print(f"Processing: {filename}.....")

        if ext in ["pdf", "png", "jpg", "jpeg"]:
            mime_type = "application/pdf" if ext == "pdf" else f"image/{ext}"
            print(f" Running Native Gemini OCR on {filename}...")
            
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            # We use gemini-2.5-flash as an on-the-fly extraction engine
            ocr_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    "Perform OCR on this file. Transcribe all text, preserving tables and hierarchy layout exactly. Return raw text only."
                ]
            )
            extracted_text = ocr_response.text
        else:
            # Plain Text Files bypass OCR
            with open(file_path, "r", encoding="utf-8") as f:
                extracted_text = f.read()

        if not extracted_text or not extracted_text.strip():
            print(f"⚠️ Warning: Could not extract readable text from {filename}.")
            continue

        # B. Fragment the resulting clean text into chunks
        file_chunks = chunk_text(extracted_text)
        print(f" Split into {len(file_chunks)} text fragments.")

        for chunk in file_chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            chunk_counter += 1

    if all_chunks:
        print("\n Bulk generating coordinates via gemini-embedding-001...")
        
        # Avoid payload limitations by submitting in batches if your file count is huge
        embed_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=all_chunks,
        )
        
        embeddings = [embedding.values for embedding in embed_response.embeddings]
        
        # Save structural fragments directly to Chroma
        collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            ids=all_ids
        )
        print(f"✅ Successfully indexed {len(all_chunks)} overall chunks to DB.")

# Run the ingestion engine
ingest_knowledge()


#3. Setup CLI Execution
print("\n RAG CLI Chatbot Initialized.")
print("Using: gemini-embedding-001 + gemini-3.5-flash")

# Initialize Google's native chat object to handle multi-turn conversational history
chat = client.chats.create(model = "gemini-3.5-flash")

while True:
    try: 
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if not user_input.strip():
            continue
        
        # A. Query local text vectors to find relevant background info
        print(" Searching Context....")
        query_embedding_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=user_input,
        )
        query_vector = query_embedding_response.embeddings[0].values

        results = collection.query(
            query_embeddings = [query_vector],
            n_results=3,
        )

        # Collapse found contexts into a single block
        retrieved_context = "\n".join(results["documents"][0]) if results["documents"] else "No relevant context found."

        # B. Construct System Prompt
        augment_prompt = (
            f"You are a helpful assistant. Use the following retrieved context to answer the user's question. "
            f"If the answer cannot be found in the context, clearly state that you do not know.\n\n"
            f"Retrieved Context:\n{retrieved_context}\n\n"
            f"User Question: {user_input}"
        )

        # C. Generate answer through Google's native multi-turn chat
        response = chat.send_message(augment_prompt)

        print(f"\nPoBot: {response.text}")

    except Exception as e:
        print(f"\nAn error occured: {e}")
        break