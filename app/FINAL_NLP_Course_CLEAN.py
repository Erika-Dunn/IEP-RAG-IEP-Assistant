# FINAL_NLP_Course_CLEAN.py

import os
import json
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ─── 1) Load FAISS index & metadata ────────────────────────────────────────────
INDEX_PATH = os.path.join('data', 'full_rag.index')
META_PATH  = os.path.join('data', 'full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta     = pickle.load(open(META_PATH, 'rb'))

# ─── 2) Embedding model for retrieval ──────────────────────────────────────────
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def vector_search(query: str, k: int = 3) -> list:
    q_vec = embedder.encode([query], normalize_embeddings=True)
    D, I  = faiss_index.search(q_vec.astype('float32'), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# ─── 3) Filtered RAG retrieval ─────────────────────────────────────────────────
def retrieve_context(goal_query: str) -> list:
    # a) Occupational context (BLS OOH only)
    occ = [h for h in vector_search(goal_query, k=3) if h['source']=='bls_ooh']
    # b) Standards context (21st Century Skills only)
    std = [h for h in vector_search("21st Century Skills " + goal_query, k=2)
           if h['source'].lower().startswith('or-21st')]
    # merge + dedupe
    merged = {h['section']: h for h in (occ + std)}
    return list(merged.values())

# ─── 4) Prompt builder with few-shot example ───────────────────────────────────
def build_prompt(profile_json: dict, docs: list[dict]) -> str:
    # Few-shot example
    example = {
      "employment_goal": "After high school, Clarence will obtain a full-time job at Walmart as a sales associate.",
      "education_goal": "After high school, Clarence will complete on-the-job training provided by Walmart and participate in employer-sponsored customer service workshops.",
      "annual_goal": "In 36 weeks, Clarence will demonstrate effective workplace communication and customer service skills by greeting customers, maintaining eye contact, listening actively, and responding to customer questions in 4 out of 5 observed opportunities.",
      "benchmarks": [
        "Greet customers appropriately",
        "Maintain eye contact",
        "Listen actively",
        "Respond to customer questions"
      ],
      "alignment": [
        "OOH standards for Retail Sales Workers",
        "21st Century Skills"
      ]
    }
    example_str = json.dumps(example, indent=2)

    # Build concise context
    ctx_lines = []
    for d in docs:
        label = "Occupational Info" if d['source']=='bls_ooh' else "Standards"
        snippet = d['text'].replace('\n',' ')[:300]
        ctx_lines.append(f"{label}: {snippet}… [SOURCE:{d['section']}]")
    context = "\n\n---\n".join(ctx_lines)

    # Final instruction
    return f"""Here is an example of the exact JSON output format I need:
{example_str}

Now, using **only** the CONTEXT below and the STUDENT PROFILE, return **only** a JSON object with exactly these five keys:
- employment_goal  
- education_goal  
- annual_goal  
- benchmarks  
- alignment  

No extra text, no code, no commentary—just the JSON.

CONTEXT:
{context}

STUDENT PROFILE (JSON):
{json.dumps(profile_json, indent=2)}
"""

# ─── 5) Zero-login LLM setup (Flan-T5) ────────────────────────────────────────────
LLM_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    temperature=0.0,
    return_full_text=False,
)

# ─── 6) LLM wrapper: parse JSON or raw fallback ─────────────────────────────────
def llm(prompt: str) -> dict:
    print("LLM PROMPT:\n", prompt)
    out = generator(prompt)[0]["generated_text"]
    print("LLM RAW OUTPUT:\n", out)
    try:
        s = out.find("{")
        e = out.rfind("}") + 1
        return json.loads(out[s:e])
    except Exception:
        return {"raw_output": out.strip()}

# ─── 7) End-to-end pipeline ─────────────────────────────────────────────────────
def process_student_profile(profile_text: str) -> dict:
    # 1) Extract structured info (no change)
    from FINAL_NLP_Course_CLEAN import extract_student_info_prompt  # if defined separately
    info = llm(extract_student_info_prompt(profile_text))

    # 2) Retrieve filtered context
    docs = retrieve_context(info.get("postsecondary_goal_employment","undecided"))

    # 3) Build prompt & generate
    prompt = build_prompt(info, docs)
    return llm(prompt)

# If you have extract_student_info_prompt separately, include it here:
def extract_student_info_prompt(profile_text: str) -> str:
    return f"""Extract the following fields as JSON (or "missing"):
- name, age, grade_level, disability, strengths,
- academic_concerns, support_needs,
- postsecondary_goal_employment, postsecondary_goal_education, postsecondary_goal_independent_living

Profile:
\"\"\"{profile_text}\"\"\"
"""
    return llm(prompt)
