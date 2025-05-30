# FINAL_NLP_Course_CLEAN.py

import os, json, pickle, faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ─── 1) Load FAISS index & metadata ─────────────────────────────────────────────
INDEX_PATH = os.path.join('data', 'full_rag.index')
META_PATH = os.path.join('data', 'full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta = pickle.load(open(META_PATH, 'rb'))

# ─── 2) Embedder for retrieval ──────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def vector_search(query: str, k: int = 3) -> list:
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(vec.astype("float32"), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# ─── 3) Load Local LLM ───────────────────────────────────────────────────────────
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# ─── 4) Prompt Template ──────────────────────────────────────────────────────────
def build_prompt(student_info: str, standards_text: str) -> str:
    return f"""You are an expert in developing IEP transition goals that are measurable, aligned to standards, and follow IDEA 2004.

Return your response ONLY as a valid JSON object that starts with {{ and matches this structure:

{{
  "employment_goal": "...",
  "education_goal": "...",
  "annual_goal": "...",
  "objectives": [
    "...",
    "...",
    "..."
  ]
}}

Student Info:
{student_info}

Relevant Career and Educational Standards:
{standards_text}
"""

# ─── 5) Generation Function ──────────────────────────────────────────────────────
def generate_iep_goals(student_info: str, retrieved_docs: list) -> dict:
    # fallback if retrieval failed
    retrieved_text = "\n".join([doc.get("text", "") for doc in retrieved_docs]).strip()
    if not retrieved_text:
        retrieved_text = (
            "Retail sales workers assist customers in finding products, handling purchases, and maintaining store presentation. "
            "They need good interpersonal and communication skills, and usually receive on-the-job training."
        )

    prompt = build_prompt(student_info, retrieved_text)

    response = llm_pipeline(prompt, max_new_tokens=512)
    result = response[0]["generated_text"].strip()

    try:
        json_start = result.find("{")
        return json.loads(result[json_start:])
    except Exception as e:
        return {
            "error": "Failed to parse LLM output as JSON.",
            "raw_output": result,
            "exception": str(e)
        }
