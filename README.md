# PoBot: Hong Kong Labour Regulations RAG CLI

PoBot is a Retrieval-Augmented Generation (RAG) command-line chatbot that answers questions about Hong Kong labour regulations using official government documents.

The system extracts text from documents, indexes them into a Chroma vector database using Google's Gemini embeddings, retrieves the most relevant document chunks for a user query, and generates an answer using Gemini while providing citations to the original document and page number.

---

# Features

- Retrieval-Augmented Generation (RAG)
- Incremental document ingestion
- PDF parsing with preserved layout and tables using PyMuPDF4LLM
- Chroma vector database
- Google Gemini Embeddings (`gemini-embedding-001`)
- Google Gemini (`gemini-3.5-flash`) for answer generation
- Maximal Marginal Relevance (MMR) retrieval
- Source and page number citations
- Retry logic for Gemini embedding rate limits

---

# Project Structure

```
.
├── google_rag_chatbot.py      # Main application entry point
├── ingest.py                  # Document ingestion and vector indexing
├── rag_pipeline.py            # Retrieval and answer generation pipeline
├── documents/                 # Knowledge base documents
├── chroma_db/                 # Chroma database
├── .env                       # Gemini API Key
├── requirements.txt
└── README.md
```

---

# System Architecture

```
                Documents
               (PDF / TXT)
                     │
                     ▼
          Document Extraction
                 PyMuPDF4LLM
                     │
                     ▼
     RecursiveCharacterTextSplitter
                     │
                     ▼
        Gemini Embedding Model
      (gemini-embedding-001)
                     │
                     ▼
        Chroma Vector Database
                     │
                     ▼
         MMR Semantic Retrieval
                     │
                     ▼
        Gemini 3.5 Flash LLM
                     │
                     ▼
      Answer + Source Citations
```

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>

cd <repository>
```

---

## 2. Create a virtual environment

macOS/Linux

```bash
python -m venv .venv

source .venv/bin/activate
```

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure your API key

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## 5. Set up Knowledge Base
Create `documents` folder and add your knowledge base documents in the `documents/` folder.

Example:

```
documents/
├── Concise Guide to the Employment Ordinance.pdf
└── Minimum Wage Guide.pdf
```

---

# Running the Project`

```bash
python google_rag_chatbot.py
```

The application will:

1. Scan the `documents/` folder.
2. Index new documents into Chroma.
3. Skip documents already indexed.
4. Launch the CLI chatbot.

Example:

```
You:
What annual leave is an employee entitled to?

Searching Context...

PoBot:

Employees employed under a continuous contract are entitled to paid annual leave after serving every period of 12 months.

Sources

- Concise Guide to the Employment Ordinance.pdf (Page 42)
```

Exit using

```
exit
```

or

```
quit
```

---

# Modules

## 1. ingest.py

Responsible for preprocessing and indexing the knowledge base.

### Responsibilities

- Scan the knowledge base folder
- Extract text from PDF files
- OCR image documents
- Split documents into semantic chunks
- Generate Gemini embeddings
- Store embeddings inside Chroma
- Store metadata
- Skip already indexed files
- Retry automatically when Gemini rate limits occur

### Stored Metadata

Each chunk stores metadata similar to:

```python
{
    "source": "Employment Ordinance.pdf",
    "page_number": 17,
    "total_pages": 90
}
```

---

## 2. rag_pipeline.py

Responsible for retrieval and answer generation.

Pipeline:

```
User Query
      │
      ▼
MMR Retrieval
      │
      ▼
Retrieve Top-k Chunks
      │
      ▼
Format Context
      │
      ▼
Gemini 3.5 Flash
      │
      ▼
Answer
      │
      ▼
Extract Citations
```

Returned object:

```python
{
    "answer": "...",
    "citations": [
        {
            "source": "...",
            "page_number": 12
        }
    ]
}
```

---

## 3. google_rag_chatbot.py

Acts as the application's entry point.

Responsibilities

- Load environment variables
- Initialize Gemini client
- Initialize Chroma
- Trigger document ingestion
- Launch the RAG pipeline
- Handle user interaction
- Display answers and citations

---

# Design Decisions

## Chroma

Chosen because it provides:

- Persistent local vector storage
- Fast similarity search
- LangChain integration
- Lightweight deployment

---

## Gemini Embeddings

The `gemini-embedding-001` model provides semantic vector representations for document chunks and integrates seamlessly with Google's ecosystem.

---

## Gemini 3.5 Flash

Selected because it offers:

- Fast inference
- Low latency
- Strong instruction following
- High-quality responses

---

## RecursiveCharacterTextSplitter

Configuration

```
Chunk Size: 1000 characters
Chunk Overlap: 200 characters
```

---

## Maximal Marginal Relevance (MMR)

Instead of plain similarity search, it uses MMR.

Benefits:

- Reduces redundant chunks
- Improves context diversity

---

## Incremental Indexing

Previously indexed documents are detected using stored metadata.

Only newly added files are embedded.

---

# Supported Document Types

- PDF
- TXT

PDF documents preserve layout and tables using PyMuPDF4LLM

---

# Source Citations

Each generated answer includes citations containing:

- Original document filename
- Page number

Example

```
Sources

- Employment Ordinance.pdf (Page 42)

- Minimum Wage Guide.pdf (Page 18)
```

---

# Error Handling

The application automatically handles:

- Missing document directory
- Empty knowledge base
- Gemini API rate limits
- Retry with exponential waiting
- Existing indexed documents

If embedding ultimately fails due to rate limits, the chatbot still initializes using the existing Chroma database.

---

# Current Limitations

- Cannot process image files.
- Modified files with the same filename are not automatically re-indexed.
- Hybrid keyword + vector retrieval is not implemented.
- Conversation history is not maintained between questions.

---

# Future Improvements

- Hybrid retrieval (BM25 + Vector Search)
- Cross-encoder reranking
- Streaming LLM responses
- Conversation memory
- Metadata filtering

---

# Dependencies

Major libraries used:

- LangChain
- ChromaDB
- LangChain Google GenAI
- PyMuPDF4LLM
- python-dotenv