"""
ingest.py — RAG Document Ingestion Pipeline
============================================
Accepts a PDF file + document type and ingests it into ChromaDB.
Can be run standalone or imported by app.py for on-the-fly ingestion.

Standalone usage:
    python ingest.py --file "brochure.pdf" --type "brochure"
    python ingest.py --file "rera_cert.pdf" --type "rera"
    python ingest.py --file "title_deed.pdf" --type "legal"
"""

import os
import re
import uuid
import argparse
import pdfplumber
import chromadb
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks
EMBED_MODEL   = "text-embedding-3-small"

# Valid document types → ChromaDB collection names
COLLECTION_MAP = {
    "brochure": "property_brochures",
    "rera":     "rera_documents",
    "legal":    "legal_documents",
}

# ── OpenAI client ─────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ============================================================
# CHROMA CLIENT — in-memory, shared across imports
# ============================================================
chroma_client = chromadb.PersistentClient(path="./chroma_store") #This saves all ingested chunks to a chroma_store/ folder on disk.

def get_collection(doc_type: str):
    """Returns (or creates) the ChromaDB collection for a doc type."""
    collection_name = COLLECTION_MAP.get(doc_type.lower())
    if not collection_name:
        raise ValueError(
            f"Unknown doc type '{doc_type}'. "
            f"Choose from: {list(COLLECTION_MAP.keys())}")
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

# ============================================================
# PDF TEXT EXTRACTION
# ============================================================
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text page by page from a PDF.
    Returns list of {page_num, text} dicts.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page_num": i + 1,
                    "text":     text.strip()
                })
    return pages


def extract_text_from_bytes(pdf_bytes: bytes) -> list[dict]:
    import io
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    from PIL import Image
    
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            # If little/no text extracted → use OCR
            if not text or len(text.strip()) < 50:
                print(f"  🔍 Page {i+1}: low text yield, trying OCR...")
                img = page.to_image(resolution=300).original
                text = pytesseract.image_to_string(img)
            
            if text and text.strip():
                pages.append({
                    "page_num": i + 1,
                    "text":     text.strip()
                })
    return pages

# ============================================================
# CHUNKING
# ============================================================
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping character-level chunks.
    Tries to split on sentence boundaries where possible.
    """
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to find a sentence boundary near the end
            boundary = text.rfind('. ', start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1  # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap  # slide back by overlap
        if start >= len(text):
            break

    return chunks

# ============================================================
# EMBEDDING
# ============================================================
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts using OpenAI text-embedding-3-small.
    Batches in groups of 100 to respect API limits.
    """
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch    = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        all_embeddings.extend([e.embedding for e in response.data])

    return all_embeddings

# ============================================================
# MAIN INGEST FUNCTION
# ============================================================
def ingest_document(
    doc_type:  str,
    filename:  str,
    pages:     list[dict],
    property_name: str = "Unknown"
) -> dict:
    """
    Core ingestion function — chunks, embeds and stores a document.

    Args:
        doc_type:      "brochure" | "rera" | "legal"
        filename:      original filename (for metadata)
        pages:         list of {page_num, text} from extraction
        property_name: optional property/project name for metadata

    Returns:
        dict with ingestion stats
    """
    collection = get_collection(doc_type)

    all_chunks    = []
    all_metadatas = []
    all_ids       = []

    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            all_chunks.append(chunk)
            all_metadatas.append({
                "doc_type":      doc_type,
                "filename":      filename,
                "page_num":      page["page_num"],
                "property_name": property_name,
            })
            all_ids.append(chunk_id)

    if not all_chunks:
        return {"status": "error", "message": "No text could be extracted from PDF."}

    print(f"  📄 Extracted {len(all_chunks)} chunks from {filename}")
    print(f"  🔢 Embedding {len(all_chunks)} chunks...")

    embeddings = embed_texts(all_chunks)

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    print(f"  ✅ Ingested {len(all_chunks)} chunks into '{COLLECTION_MAP[doc_type]}'")

    return {
        "status":     "success",
        "doc_type":   doc_type,
        "filename":   filename,
        "chunks":     len(all_chunks),
        "pages":      len(pages),
        "collection": COLLECTION_MAP[doc_type],
    }

# ============================================================
# RETRIEVAL
# ============================================================
def retrieve_chunks(
    query:    str,
    doc_type: str,
    n_results: int = 5
) -> list[dict]:
    """
    Retrieves the most relevant chunks for a query from a collection.
    Returns list of {text, metadata, distance} dicts.
    """
    collection = get_collection(doc_type)

    # Check collection isn't empty
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":     doc,
            "metadata": meta,
            "distance": round(dist, 4),
        })

    return chunks


def retrieve_from_all(query: str, n_results: int = 5) -> list[dict]:
    """
    Retrieves from ALL collections and merges results by relevance.
    Used when doc_type is not specified.
    """
    all_chunks = []
    for doc_type in COLLECTION_MAP.keys():
        try:
            chunks = retrieve_chunks(query, doc_type, n_results=3)
            all_chunks.extend(chunks)
        except Exception:
            continue

    # Sort by distance (lower = more similar)
    all_chunks.sort(key=lambda x: x["distance"])
    return all_chunks[:n_results]

# ============================================================
# COLLECTION STATUS
# ============================================================
def get_collection_stats() -> dict:
    """Returns count of documents in each collection."""
    stats = {}
    for doc_type, collection_name in COLLECTION_MAP.items():
        try:
            col = chroma_client.get_or_create_collection(collection_name)
            stats[doc_type] = col.count()
        except Exception:
            stats[doc_type] = 0
    return stats

# ============================================================
# STANDALONE CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB")
    parser.add_argument("--file", required=True, help="Path to PDF file")
    parser.add_argument("--type", required=True,
                        choices=["brochure", "rera", "legal"],
                        help="Document type")
    parser.add_argument("--property", default="Unknown",
                        help="Property/project name (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        exit(1)

    print(f"\n🔍 Extracting text from: {args.file}")
    pages = extract_text_from_pdf(args.file)
    print(f"   Found {len(pages)} pages with text")

    result = ingest_document(
        doc_type=args.type,
        filename=os.path.basename(args.file),
        pages=pages,
        property_name=args.property
    )

    if result["status"] == "success":
        print(f"\n✅ Done — {result['chunks']} chunks stored in '{result['collection']}'")
    else:
        print(f"\n❌ Error: {result['message']}")