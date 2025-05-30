# FINAL_NLP_Course_CLEAN.py

import os, json, pickle, faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

# ─── 1) Load FAISS Index & Metadata ─────────────────────────────────────────────
INDEX_PATH = os.path.join('data','full_rag.index')
META_PATH  = os.path.join('data','full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta    = pickle.load(open(META_PATH, 'rb'))

# ─── 2) Sentence Embedding Model ────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def vector_search(query: str, k: int = 3) -> list:
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(vec.astype("float32"), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# ─── 3) Load Local LLM ──────────────────────────────────────────────────────────
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

# ─── 4) Prompt Template ─────────────────────────────────────────────────────────
def build_prompt(student_info: str, retrieved_docs: list) -> str:
    retrieved_text = "\n\n".join(doc.get("content", "")[:1000] for doc in retrieved_docs[:2]).strip()
    if not retrieved_text:
        retrieved_text = (
            "Retail sales workers assist customers, handle purchases, and maintain store presentation. "
            "They need interpersonal and communication skills and typically receive on-the-job training."
        )

    return f"""
You are an expert in developing IEP transition goals that are measurable, aligned to standards, and follow IDEA 2004.

Return ONLY valid JSON (no explanations). The JSON must contain **exactly one** key for each of the following:

- "employment_goal"
- "education_goal"
- "annual_goal"
- "objectives" → this must be a list of 3 unique strings.

If unsure, make a professional assumption. Here's the expected format:

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

### Student Profile:
{student_info}

### Career and Educational Standards:
{retrieved_text}
"""

# ─── 5) Goal Generator ──────────────────────────────────────────────────────────
def generate_iep_goals(student_info: str, retrieved_docs: list) -> dict:
    prompt = build_prompt(student_info, retrieved_docs)

    try:
        response = llm_pipeline(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"].strip()

        # TEMP debug print:
        print("==== RAW OUTPUT ====")
        print(response[:2000])  # Only print first 2k chars
        print("====================")

        if not response.startswith("{"):
            response = "{" + response.split("{", 1)[-1]

        json_start = response.find("{")
        json_end = response.rfind("}")
        json_str = response[json_start:json_end + 1]

        parsed = json.loads(json_str)

        if isinstance(parsed.get("objectives"), str):
            parsed["objectives"] = [parsed["objectives"]]

        parsed["objectives"] = list(dict.fromkeys(parsed.get("objectives", [])))[:3]

        return parsed
    except Exception as e:
        return {
            "error": "Failed to parse LLM output as JSON.",
            "raw_output": response,
            "exception": str(e)
        }
