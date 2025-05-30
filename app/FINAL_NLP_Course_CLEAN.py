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

Return your response as a JSON object starting with {{, with no extra text or formatting.

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
# Goal generator with HTML cleanup and JSON fallback
# ─────────────────────────────────────────────────────────────────────────────

def generate_iep_goals(student_info: str, retrieved_docs: list) -> dict:
    # Filter irrelevant content (optional: add more advanced logic here)
    relevant_docs = [doc for doc in retrieved_docs if "retail" in doc.get("text", "").lower()]
    if not relevant_docs:
        relevant_docs = retrieved_docs

    # Clean up HTML and join
    retrieved_text = "\n".join([strip_html(doc.get("content", "")) for doc in relevant_docs]).strip()

    if not retrieved_text:
        retrieved_text = (
            "No specific career or educational standards were retrieved. "
            "Use the student's profile and make professional assumptions about appropriate job skills and training needs based on typical entry-level employment in their interest area."
    )


    # Build prompt
    prompt = build_prompt(student_info, retrieved_text)

    # Run the LLM
    response = llm_pipeline(prompt, max_new_tokens=1024, do_sample=False, temperature=0.3)
    result = response[0].get("generated_text", "").strip()

    if not result:
        return {
            "error": "Model returned no output.",
            "prompt": prompt[:1500],
            "retrieved_text": retrieved_text[:1000]
        }

    try:
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

