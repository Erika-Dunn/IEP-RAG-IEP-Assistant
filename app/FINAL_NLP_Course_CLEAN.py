# FINAL_NLP_Course_CLEAN.py

import os
import json
import pickle
import faiss
import torch
import re
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

Return your response ONLY as a raw JSON object (not a Python list or string). Your output must start with `{{` and be fully parsable by `json.loads()`.

Here is the required JSON structure:

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

If any detail is missing, make a professional assumption.

### Student Profile:
{student_info}

### Career and Educational Standards:
{retrieved_chunks}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Goal generator with JSON recovery fallback
# ─────────────────────────────────────────────────────────────────────────────

def generate_iep_goals(student_info: str, retrieved_docs: list) -> dict:
    retrieved_text = "\n".join([doc.get("content", "") for doc in retrieved_docs])
    prompt = build_prompt(student_info, retrieved_text)

    response = llm_pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
    result = response[0]["generated_text"].strip()

    try:
        # Try to extract the first JSON-like block
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError("No valid JSON object found in output.")
    except Exception as e:
        return {
            "error": "Failed to parse LLM output as JSON.",
            "raw_output": result,
            "exception": str(e)
        }
