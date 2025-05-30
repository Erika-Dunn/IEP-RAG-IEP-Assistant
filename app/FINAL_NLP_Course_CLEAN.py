# FINAL_NLP_Course_CLEAN.py

import os, json, pickle, faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ─── 1. Load FAISS Index and Metadata ────────────────────────────────────────────
INDEX_PATH = os.path.join('data','full_rag.index')
META_PATH  = os.path.join('data','full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta    = pickle.load(open(META_PATH,'rb'))

# ─── 2. Embedder for Retrieval ───────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def vector_search(query: str, k: int = 3) -> list:
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(vec.astype("float32"), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# ─── 3. Load Local LLM ───────────────────────────────────────────────────────────
model_name = "google/flan-t5-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

# ─── 4. Prompt Constructor ───────────────────────────────────────────────────────
def build_prompt(student_info: str, retrieved_text: str) -> str:
    return f"""
You are an expert in developing IEP transition goals that are measurable, aligned to standards, and follow IDEA 2004.

Return your response ONLY as a raw JSON object (not a Python list or string). Your output must start with {{ and be fully parsable by `json.loads()`.

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
{retrieved_text}
""".strip()

# ─── 5. IEP Generation Function ─────────────────────────────────────────────────
def generate_iep_goals(student_info: str, retrieved_docs: list) -> dict:
    retrieved_text = "\n".join([doc.get("content", "") for doc in retrieved_docs])
    prompt = build_prompt(student_info, retrieved_text)

    try:
        response = llm_pipeline(prompt, max_new_tokens=1024)
        result = response[0]["generated_text"].strip()
        if not result.startswith("{"):
            return {
                "error": "Model returned no output.",
                "prompt": prompt,
                "retrieved_text": retrieved_text,
                "raw_output": result
            }
        return json.loads(result)
    except Exception as e:
        return {
            "error": "Failed to parse LLM output as JSON.",
            "raw_output": result if 'result' in locals() else "",
            "exception": str(e)
        }
