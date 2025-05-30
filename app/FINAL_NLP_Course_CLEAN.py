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

# ─── 4) Prompt Constructor ───────────────────────────────────────────────────────
def build_prompt(student_info: str, retrieved_docs: list) -> str:
    retrieved_text = "\n\n".join(doc.get("content", "")[:1000] for doc in retrieved_docs[:2]).strip()
    if not retrieved_text:
        retrieved_text = (
            "Retail sales workers assist customers, handle purchases, and maintain store presentation. "
            "They need interpersonal and communication skills and typically receive on-the-job training."
        )

    return f"""You are an expert in developing IEP transition goals that are measurable, aligned to standards, and follow IDEA 2004.

Return your response as a JSON object starting with {{ and using ONLY the following keys:

{{
  "employment_goal": "Clarence will obtain a full-time job at Walmart.",
  "education_goal": "Clarence will complete on-the-job training after high school.",
  "annual_goal": "Clarence will demonstrate customer service skills in a retail simulation.",
  "objectives": [
    "Greet customers appropriately during role-play activities.",
    "Maintain eye contact and respond to customer questions.",
    "Complete a customer interaction checklist with 90% accuracy."
  ]
}}

The `objectives` field must be a JSON array with exactly 3 items. Use professional assumptions if data is missing.

### Student Profile:
{student_info}

### Career and Educational Standards:
{retrieved_text}
"""

# ─── 5) Generate Structured IEP Goals ────────────────────────────────────────────
def generate_iep_goals(student_info: str, retrieved_docs: list) -> dict:
    prompt = build_prompt(student_info, retrieved_docs)

    try:
        raw = llm_pipeline(prompt + "\n{", max_new_tokens=512, do_sample=False)[0]["generated_text"].strip()
        response = "{" + raw if not raw.strip().startswith("{") else raw
    except Exception as e:
        return {
            "error": "Model returned no output or failed.",
            "prompt": prompt[:1000],
            "exception": str(e)
        }

    try:
        json_start = response.find("{")
        parsed = json.loads(response[json_start:])

        # Postprocess malformed objective strings
        if isinstance(parsed.get("objectives"), str):
            parsed["objectives"] = [parsed["objectives"]]

        if "objectives" in parsed and isinstance(parsed["objectives"], list):
            parsed["objectives"] = parsed["objectives"][:3]

        return parsed
    except Exception as e:
        return {
            "error": "Failed to parse LLM output as JSON.",
            "raw_output": response,
            "exception": str(e)
        }

