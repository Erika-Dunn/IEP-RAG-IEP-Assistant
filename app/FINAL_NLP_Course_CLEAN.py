# FINAL_NLP_Course_CLEAN.py

import os
import json
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────────────────────────────────────────────────────────
# Load FAISS index and metadata
# ─────────────────────────────────────────────────────────────────────────────

INDEX_PATH = os.path.join("data", "full_rag.index")
META_PATH = os.path.join("data", "full_rag_metadata.pkl")

faiss_index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    doc_meta = pickle.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Embedder for semantic search
# ─────────────────────────────────────────────────────────────────────────────

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def vector_search(query: str, k: int = 3) -> list:
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(vec.astype("float32"), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# ─────────────────────────────────────────────────────────────────────────────
# Load local Flan-T5 model for text generation
# ─────────────────────────────────────────────────────────────────────────────

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(student_info: str, retrieved_chunks: str) -> str:
    return f"""You are an expert in developing IEP transition goals that are measurable, aligned to standards, and follow IDEA 2004.

Given the student's profile and relevant information from career and education standards, create the following in **JSON** format:

{{
  "employment_goal": "Measurable postsecondary employment goal.",
  "education_goal": "Measurable postsecondary education/training goal.",
  "annual_goal": "Annual IEP goal aligned to state standards.",
  "objectives": [
    "Short-term objective 1 supporting the annual goal.",
    "Short-term objective 2 supporting the annual goal.",
    "Short-term objective 3 supporting the annual goal."
  ]
}}

If required information is missing or ambiguous, make a professional assumption and include it.

### Student Profile:
{student_info}

### Career and Educational Standards:
{retrieved_chunks}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Generation function used in Gradio interface
# ─────────────────────────────────────────────────────────────────────────────

def generate_iep_goals(student_info: str, retrieved_docs: list) -> dict:
    retrieved_text = "\n".join([doc.get("content", "") for doc in retrieved_docs])
    prompt = build_prompt(student_info, retrieved_text)

    response = llm_pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
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
