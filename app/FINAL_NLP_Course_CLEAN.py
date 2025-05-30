import os
import json
import pickle
import faiss
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# ─── 1. Load FAISS index & metadata ───────────────────────────────────────────
INDEX_PATH = os.path.join("data", "full_rag.index")
META_PATH = os.path.join("data", "full_rag_metadata.pkl")
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta = pickle.load(open(META_PATH, "rb"))

# ─── 2. Embedder for retrieval ────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
def vector_search(query: str, k: int = 3) -> list:
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(vec.astype("float32"), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# ─── 3. Load HuggingFace model pipeline ───────────────────────────────────────
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

# ─── 4. LLM wrapper ───────────────────────────────────────────────────────────
def llm(prompt: str) -> str:
    try:
        response = pipe(
            prompt,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
        )
        return response[0]["generated_text"].strip()
    except Exception as e:
        return json.dumps({"error": "Model failed to generate output.", "exception": str(e)})

# ─── 5. Goal generation ────────────────────────────────────────────────────────
def generate_iep_goals(student_profile: str, retrieved_docs: list) -> dict:
    # Construct knowledge base text
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])

    # Prompt engineering
    prompt = f"""
You are an expert in developing IEP transition goals that are measurable, aligned to standards, and follow IDEA 2004.

Return your response ONLY as a raw JSON object (not a Python list or string). Your output must start with {{ and be fully parsable by json.loads().

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
{student_profile}

### Career and Educational Standards:
{context}
"""

    raw_output = llm(prompt)

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse LLM output as JSON.",
            "raw_output": raw_output,
        }
