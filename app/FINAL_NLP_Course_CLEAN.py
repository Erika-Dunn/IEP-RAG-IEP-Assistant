# FINAL_NLP_Course_CLEAN.py

import os
import json
import pickle
import faiss
import torch
import re
from bs4 import BeautifulSoup
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
# HTML cleaner
# ─────────────────────────────────────────────────────────────────────────────

def strip_html(text):
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(student_info: str, retrieved_chunks: str) -> str:
    return f"""You are an expert in developing IEP transition goals that are measurable, aligned to standards, and follow IDEA 2004.

Your response MUST be a complete and valid JSON object. Do not include any explanation, markdown, or extra characters. Only return the JSON. Begin with '{{' and match the structure below exactly.

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
# Goal generator with HTML cleanup and JSON fallback
# ─────────────────────────────────────────────────────────────────────────────

def generate_iep_goals(student_info: str, retrieved_docs: list) -> dict:
    # Clean up retrieved text
    clean_chunks = [strip_html(doc.get("content", "")) for doc in retrieved_docs]
    retrieved_text = "\n".join(chunk for chunk in clean_chunks if chunk.strip()).strip()

    # Fallback if nothing useful retrieved
    if not retrieved_text:
        retrieved_text = (
            "No specific career or educational standards were retrieved. "
            "Use the student's profile and make professional assumptions about appropriate job skills and training needs based on typical entry-level employment in their interest area."
        )

    # Build prompt
    prompt = build_prompt(student_info, retrieved_text)

    # Run generation
    try:
        response = llm_pipeline(prompt, max_new_tokens=512)
        result = response[0].get("generated_text", "").strip()

        if not result or len(result) < 20:
            return {
                "error": "LLM output was too short or empty.",
                "raw_output": result,
                "prompt": prompt[:1000]
            }

        # Extract JSON
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            normalized = {k.lower(): v for k, v in parsed.items()}
            return normalized
        else:
            raise ValueError("No valid JSON object found.")
    except Exception as e:
        return {
            "error": "Failed to parse LLM output as JSON.",
            "raw_output": result if 'result' in locals() else "",
            "exception": str(e)
        }
