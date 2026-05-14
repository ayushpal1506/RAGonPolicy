"""
Policy RAG — FastAPI Backend
"""

import os, uuid, json, shutil
import anthropic

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from rag_engine import (
    ingest_pdf, retrieve_chunks, build_prompt,
    get_store, list_documents, delete_document
)

app = FastAPI(title="Policy RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

store            = get_store()
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

TASK_QUERIES = {
    "summary":        "overview purpose scope coverage what does this policy cover",
    "risks":          "risk warning liability danger penalty consequence",
    "exclusions":     "excluded not covered exception does not apply",
    "key_limits":     "limit cap maximum amount threshold monetary value",
    "obligations":    "must shall required obligation duty policyholder",
    "claims_process": "claim procedure deadline document submit notification",
}

TASK_PROMPTS = {
    "summary": (
        "Provide a concise summary of this policy in 4-6 sentences covering: "
        "what it covers, who it applies to, and its main purpose. "
        'Return JSON: {"summary": "...", "citations": [{"page": N, "section": "..."}]}'
    ),
    "risks": (
        "Identify ALL risks, warnings, and potential liabilities mentioned. "
        'Return JSON: {"risks": [{"risk": "...", "severity": "high/medium/low", "page": N, "section": "..."}]}'
    ),
    "exclusions": (
        "List EVERY exclusion, exception, or clause stating what is NOT covered. "
        'Return JSON: {"exclusions": [{"exclusion": "...", "page": N, "section": "..."}]}'
    ),
    "key_limits": (
        "Extract ALL monetary limits, time limits, caps, and thresholds. "
        'Return JSON: {"limits": [{"limit": "...", "value": "...", "page": N, "section": "..."}]}'
    ),
    "obligations": (
        "List all obligations, duties, and requirements placed on the policyholder. "
        'Return JSON: {"obligations": [{"obligation": "...", "page": N, "section": "..."}]}'
    ),
    "claims_process": (
        "Describe the claims procedure step by step, including deadlines and required documents. "
        'Return JSON: {"claims_steps": [{"step": "...", "deadline": "...", "page": N, "section": "..."}]}'
    ),
}


class AnalyzeRequest(BaseModel):
    doc_id: str
    tasks: list = ["summary", "risks", "exclusions", "key_limits"]

class AskRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 6


def call_llm(prompt: str) -> dict:
    if not anthropic_client.api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    try:
        msg = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {"raw_response": raw, "parse_error": str(e)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "docs_ingested": len(list_documents(store))}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")
    doc_id    = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = ingest_pdf(save_path, doc_id, store)
        result["filename"] = file.filename
        return result
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    results = {}
    for task_key in req.tasks:
        prompt_text = TASK_PROMPTS.get(task_key)
        if not prompt_text:
            results[task_key] = {"error": f"Unknown task '{task_key}'"}
            continue
        query  = TASK_QUERIES.get(task_key, task_key)
        chunks = retrieve_chunks(query, req.doc_id, store, top_k=6)
        if not chunks:
            results[task_key] = {"error": "No relevant content found."}
            continue
        prompt = build_prompt(prompt_text, chunks)
        results[task_key] = call_llm(prompt)
    return {"doc_id": req.doc_id, "results": results}

@app.post("/ask")
def ask(req: AskRequest):
    chunks = retrieve_chunks(req.question, req.doc_id, store, top_k=req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found.")
    task = (
        f"Answer this question based only on the context: {req.question}\n\n"
        'Return JSON: {"answer": "...", "confidence": "high/medium/low", '
        '"citations": [{"page": N, "section": "..."}]}'
    )
    llm_result = call_llm(build_prompt(task, chunks))
    return {
        "question":    req.question,
        "answer":      llm_result,
        "chunks_used": len(chunks),
        "source_pages": sorted(set(c["page"] for c in chunks))
    }

@app.get("/documents")
def get_documents():
    docs = list_documents(store)
    details = []
    for d in docs:
        info = store.doc_info(d)
        details.append({"doc_id": d, **(info or {})})
    return {"documents": details}

@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    if not delete_document(doc_id, store):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")