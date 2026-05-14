"""
Policy RAG Engine
-----------------
Handles: PDF extraction → section detection → chunking → TF-IDF embedding → retrieval

Uses sklearn TF-IDF instead of neural embeddings — works offline, fast, no model downloads,
and performs very well on legal/policy text with its precise terminology.
"""

import fitz  # PyMuPDF
import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VECTORDB_DIR = "./vectordb"
os.makedirs(VECTORDB_DIR, exist_ok=True)


class VectorStore:
    def __init__(self, store_dir: str = VECTORDB_DIR):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self._cache: dict = {}

    def _path(self, doc_id: str) -> str:
        return os.path.join(self.store_dir, f"{doc_id}.pkl")

    def save(self, doc_id: str, chunks: list) -> int:
        texts = [c["text"] for c in chunks]
        vec   = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
        mat   = vec.fit_transform(texts)
        entry = {"vectorizer": vec, "matrix": mat, "chunks": chunks}
        with open(self._path(doc_id), "wb") as f:
            pickle.dump(entry, f)
        self._cache[doc_id] = entry
        return len(chunks)

    def load(self, doc_id: str):
        if doc_id in self._cache:
            return self._cache[doc_id]
        path = self._path(doc_id)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            entry = pickle.load(f)
        self._cache[doc_id] = entry
        return entry

    def query(self, doc_id: str, query: str, top_k: int = 6) -> list:
        entry = self.load(doc_id)
        if not entry:
            return []
        q_vec   = entry["vectorizer"].transform([query])
        scores  = cosine_similarity(q_vec, entry["matrix"])[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] < 0.01:
                continue
            c = entry["chunks"][idx]
            results.append({
                "text":    c["text"],
                "page":    c["page_num"],
                "section": c["section"],
                "score":   round(float(scores[idx]), 3)
            })
        return results

    def list_docs(self) -> list:
        return [f[:-4] for f in os.listdir(self.store_dir) if f.endswith(".pkl")]

    def delete(self, doc_id: str) -> bool:
        path = self._path(doc_id)
        if os.path.exists(path):
            os.remove(path)
            self._cache.pop(doc_id, None)
            return True
        return False

    def doc_info(self, doc_id: str):
        entry = self.load(doc_id)
        if not entry:
            return None
        chunks = entry["chunks"]
        return {
            "total_chunks":   len(chunks),
            "total_pages":    max(c["page_num"] for c in chunks),
            "sections_found": sorted(set(c["section"] for c in chunks))
        }


_store = VectorStore(VECTORDB_DIR)

def get_store() -> VectorStore:
    return _store


def inspect_fonts(pdf_path: str) -> dict:
    doc   = fitz.open(pdf_path)
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))
    doc.close()
    if not sizes:
        return {}
    from collections import Counter
    counts    = Counter(sizes)
    body_size = counts.most_common(1)[0][0]
    return {
        "all_sizes":                   sorted(set(sizes)),
        "body_size":                   body_size,
        "suggested_heading_threshold": body_size + 1.5
    }


def extract_with_sections(pdf_path: str, heading_threshold: Optional[float] = None) -> list:
    doc       = fitz.open(pdf_path)
    font_info = inspect_fonts(pdf_path)
    if heading_threshold is None:
        heading_threshold = font_info.get("suggested_heading_threshold", 13.0)

    current_section = "Preamble"
    spans_out       = []

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    is_bold = "bold" in span["font"].lower()
                    if is_bold or span["size"] >= heading_threshold:
                        current_section = text
                    else:
                        spans_out.append({
                            "text":     text,
                            "page_num": page_num,
                            "section":  current_section
                        })
    doc.close()
    return spans_out


def chunk_with_metadata(spans: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    grouped = defaultdict(list)
    for s in spans:
        grouped[(s["page_num"], s["section"])].append(s["text"])

    all_chunks = []
    for (page_num, section), lines in grouped.items():
        for split in splitter.split_text(" ".join(lines)):
            if len(split.strip()) < 30:
                continue
            all_chunks.append({
                "text":        split.strip(),
                "page_num":    page_num,
                "section":     section,
                "chunk_index": len(all_chunks)
            })
    return all_chunks


def ingest_pdf(pdf_path: str, doc_id: str, store: VectorStore,
               chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
    spans  = extract_with_sections(pdf_path)
    chunks = chunk_with_metadata(spans, chunk_size, chunk_overlap)
    n      = store.save(doc_id, chunks)
    return {
        "doc_id":         doc_id,
        "total_chunks":   n,
        "total_pages":    max(c["page_num"] for c in chunks) if chunks else 0,
        "sections_found": sorted(set(c["section"] for c in chunks))
    }


def retrieve_chunks(query: str, doc_id: str, store: VectorStore, top_k: int = 6) -> list:
    return store.query(doc_id, query, top_k)


def build_prompt(task: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(
        f"[Page {c['page']} | Section: {c['section']}]\n{c['text']}"
        for c in chunks
    )
    return f"""You are a policy analysis assistant.
Use ONLY the context provided below. Do NOT guess or infer beyond what is stated.
For every point you make, cite the exact Page and Section from the context.
If information is not found in the context, say "Not found in document."

=== CONTEXT ===
{context}

=== TASK ===
{task}

Respond in valid JSON only. No markdown, no explanation outside the JSON.
"""


def list_documents(store: VectorStore) -> list:
    return store.list_docs()

def delete_document(doc_id: str, store: VectorStore) -> bool:
    return store.delete(doc_id)