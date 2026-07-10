import glob
import os

import chromadb
import pymupdf4llm
from google import genai
from google.genai import types

KNOWLEDGE_DIR = "./documents"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "google_rag_knowledge"
EMBED_BATCH_SIZE = 100
EMBEDDING_MODEL = "gemini-embedding-001"
OCR_MODEL = "gemini-2.5-flash"

SUPPORTED_EXTENSIONS = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.txt"]


def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text, max_chars=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks


def get_indexed_sources(collection):
    if collection.count() == 0:
        return set()

    existing = collection.get(include=["metadatas"])
    return {
        meta["source"]
        for meta in existing["metadatas"]
        if meta and "source" in meta
    }


def extract_pdf_text(file_path):
    print(" Extracting with pymupdf4llm (layout + tables)...")
    return pymupdf4llm.to_markdown(
        file_path,
        page_separators=True,
        show_progress=True,
    )


def extract_image_text(client, file_path, ext):
    mime_type = f"image/{ext}"
    print(" Running Gemini OCR on image...")
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    ocr_response = client.models.generate_content(
        model=OCR_MODEL,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            "Perform OCR on this file. Transcribe all text, preserving tables and hierarchy layout exactly. Return raw text only.",
        ],
    )
    return ocr_response.text


def embed_in_batches(client, chunks):
    embeddings = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        embed_response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
        )
        embeddings.extend(embedding.values for embedding in embed_response.embeddings)
    return embeddings


def ingest_knowledge(client, collection):
    print(f"Scanning '{KNOWLEDGE_DIR}' directory for files...")
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        print(f"Created empty '{KNOWLEDGE_DIR}' folder. Drop PDFs or images inside and re-run.")
        return 0

    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(glob.glob(os.path.join(KNOWLEDGE_DIR, ext)))

    if not all_files:
        print(f"No valid documents found in '{KNOWLEDGE_DIR}'. Add files to begin.")
        return 0

    indexed_sources = get_indexed_sources(collection)
    pending_files = [
        file_path
        for file_path in all_files
        if os.path.basename(file_path) not in indexed_sources
    ]

    if not pending_files:
        print(f"All {len(all_files)} file(s) already indexed. Delete '{CHROMA_PATH}' to re-ingest.")
        return 0

    all_chunks = []
    all_ids = []
    all_metadatas = []
    chunk_counter = collection.count()

    for file_path in pending_files:
        filename = os.path.basename(file_path)
        ext = filename.split(".")[-1].lower()

        print(f"Processing: {filename}...")

        if ext == "pdf":
            extracted_text = extract_pdf_text(file_path)
        elif ext in ["png", "jpg", "jpeg"]:
            extracted_text = extract_image_text(client, file_path, ext)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                extracted_text = f.read()

        if not extracted_text or not extracted_text.strip():
            print(f"Warning: Could not extract readable text from {filename}.")
            continue

        file_chunks = chunk_text(extracted_text)
        print(f" Split into {len(file_chunks)} text fragments.")

        for chunk_index, chunk in enumerate(file_chunks):
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metadatas.append({
                "source": filename,
                "chunk_index": chunk_index,
            })
            chunk_counter += 1

    if not all_chunks:
        return 0

    print(f"\nGenerating embeddings via {EMBEDDING_MODEL}...")
    embeddings = embed_in_batches(client, all_chunks)

    collection.add(
        embeddings=embeddings,
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas,
    )
    print(f"Successfully indexed {len(all_chunks)} new chunks to DB.")
    return len(all_chunks)
