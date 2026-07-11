import glob
import os
import re
import time

import pymupdf4llm
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


KNOWLEDGE_DIR = "./documents"
COLLECTION_NAME = "google_rag_knowledge"
EMBEDDING_MODEL = "gemini-embedding-001"
OCR_MODEL = "gemini-2.5-flash"
EMBED_BATCH_SIZE = 20
BATCH_DELAY_SECONDS = 15
MAX_RETRIES = 5
CHROMA_PATH = "./chroma_db"

SUPPORTED_EXTENSIONS = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.txt"]

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME
    )


def get_indexed_sources(vectorstore):

    data = vectorstore.get(include=["metadatas"])
    if not data["metadatas"]:
        return set()
    sources = set ()
    for meta in data["metadatas"]:
        if meta and "source" in meta:
            sources.add(os.path.basename(str(meta["source"])))
    print(f"Found {len(sources)} indexed sources.")        
    return sources


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


def load_document(client, file_path):
    filename = os.path.basename(file_path)
    ext = filename.split(".")[-1].lower()

    if ext == "pdf":
        text = extract_pdf_text(file_path)
    elif ext in ["png", "jpg", "jpeg"]:
        text = extract_image_text(client, file_path, ext)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text or not text.strip():
        return []

    return [Document(page_content=text, metadata={"source": str(filename)})]


def parse_retry_delay(error_message):
    match = re.search(r"retry in ([0-9.]+)s", error_message, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1
    return None


def add_documents_with_retry(vectorstore, documents):
    for start in range(0, len(documents), EMBED_BATCH_SIZE):
        batch = documents[start : start + EMBED_BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as error:
                if "429" not in str(error) and "RESOURCE_EXHAUSTED" not in str(error):
                    raise

                wait_seconds = parse_retry_delay(str(error)) or BATCH_DELAY_SECONDS * (attempt + 1)
                print(f" Rate limited. Waiting {wait_seconds:.0f}s before retry...")
                time.sleep(wait_seconds)
        else:
            raise RuntimeError("Embedding failed after maximum retries due to rate limits.")

        if start + EMBED_BATCH_SIZE < len(documents):
            time.sleep(BATCH_DELAY_SECONDS)


def ingest_knowledge(client, vectorstore):
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

    indexed_sources = get_indexed_sources(vectorstore)
    pending_files = [
        file_path
        for file_path in all_files
        if os.path.basename(file_path) not in indexed_sources
    ]

    if not pending_files:
        print(f"All {len(all_files)} file(s) already indexed in Chroma.")
        return 0

    total_chunks = 0

    for file_path in pending_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}...")

        documents = load_document(client, file_path)
        if not documents:
            print(f"Warning: Could not extract readable text from {filename}.")
            continue

        chunks = TEXT_SPLITTER.split_documents(documents)
        print(f" Split into {len(chunks)} text fragments.")
        print(f" Embedding via {EMBEDDING_MODEL} (batches of {EMBED_BATCH_SIZE})...")

        add_documents_with_retry(vectorstore, chunks)
        total_chunks += len(chunks)
        print(f" Indexed {len(chunks)} chunks from {filename}.")

    if total_chunks:
        print(f"Successfully indexed {total_chunks} new chunks to DB.")

    return total_chunks
