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

SUPPORTED_EXTENSIONS = ["*.pdf", "*.txt"]

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
    """
    Retrieves the list of sources already indexed in the vectorstore.
    Prevents processing and indexing the same documents multiple times.
    Args:
        vectorstore: The Chroma vectorstore.

    Returns:
        set: A set of indexed sources.
    """
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
    """
    Extract text from a PDF file.
    Args:
        file_path: The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    print(" Extracting with pymupdf4llm (layout + tables)...")
    return pymupdf4llm.to_markdown(
        file_path,
        page_separators=True,
        show_progress=True,
    )


def load_document(file_path):
    """
    Load a document from the given file path.
    Args:
        file_path: The path to the document.

    Returns:
        List of Document objects.
    """
    filename = os.path.basename(file_path)
    ext = filename.split(".")[-1].lower()

    if ext == "pdf":
        text = extract_pdf_text(file_path)

        #Extract page number for pymupdf4llm's page separators: e.g. "--- end of page 1 ---"
        pages = re.split(
            r"--- end of page.*page_number=\d+ ---",
            text,
        )

        documents = []

        for page_number, page_text in enumerate(pages, start=1):
            if not page_text:
                continue

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": filename,
                        "page_number": page_number,
                        "total_pages": len(pages),
                    }
                )
            )

    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text or not text.strip():
        return []

    return documents

def parse_retry_delay(error_message):
    #Parse retry delay from error message for rate limits
    match = re.search(r"retry in ([0-9.]+)s", error_message, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1
    return None


def add_documents_with_retry(vectorstore, documents):
    """
    Add documents to the vectorstore with retry logic to handle rate limits.
    Args:
        vectorstore: The Chroma vectorstore.
        documents: List of documents to add.

    Returns:
        bool: True if documents were added successfully.
    """

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
    return True

def ingest_knowledge(client, vectorstore):
    """
    Args:
        client: The Google Generative AI client.
        vectorstore: The Chroma vectorstore.

    Returns:
        int: The number of chunks indexed.
    """

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

        documents = load_document(file_path)
        if not documents:
            print(f"Warning: Could not extract readable text from {filename}.")
            continue

        chunks = TEXT_SPLITTER.split_documents(documents)
        print(f" Split into {len(chunks)} text fragments.")
        print(f" Embedding via {EMBEDDING_MODEL} (batches of {EMBED_BATCH_SIZE})...")

        success = add_documents_with_retry(vectorstore, chunks)
        if not success:
            print(f"Rate limit exceeded while indexing {filename}. Continuing...")
            continue

        total_chunks += len(chunks)
        print(f" Indexed {len(chunks)} chunks from {filename}.")

    if total_chunks:
        print(f"Successfully indexed {total_chunks} new chunks to DB.")

    return total_chunks
