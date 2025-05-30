# FINAL_NLP_Course_CLEAN.py

import os
import json
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ── 1) Load FAISS index & metadata ─────────────────────────────────────────────
INDEX_PATH = os.path.join('data', 'full_rag.index')
META_PATH  = os.path.join('data', 'full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta     = pickle.load(open(META_PATH, 'rb'))

# ── 2) Embedding model ─────────────────────────────────────────────────────────
embedder = SentenceTransformer('all-MiniLM-L6-v2')
def vector_search(query: str, k: int = 3) -> list:
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(vec.astype('float32'), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# ── 3) Filtered RAG retrieval ───────────────────────────────────────────────────
def retrieve_context(goal_query: str) -> list:
    # Occupational (BLS OOH)
    occ = [h for h in vector_search(goal_query, k=3) if h['source']=='bls_ooh']
    # Standards (21st Century)
    std = [h for h in vector_search("21st Century Skills " + goal_query, k=2)
           if h['source'].lower().startswith('or-21st')]
    merged = {h['section']: h for h in (occ + std)}
    return list(merged.values())

# ── 4) Prompt builders ───────────────────────────────────────────────────────────
def extract_student_info_prompt(profile_text: str) -> str:
    return f"""Extract the following fields as JSON from the student's profile below.
If missing, use "missing". Keys:
name, age, grade_level, disability, strengths,
academic_concerns, support_needs,
postsecondary_goal_employment,
postsecondary_goal_education,
postsecondary_goal_independent_living

Profile:
\"\"\"{profile_text}\"\"\"
"""

def generate_goals_prompt(profile_json: dict) -> str:
    return f"""Based ONLY on the JSON profile below, return exactly a JSON object with keys:
- employment_goal
- education_goal
- annual_goal
- benchmarks
- alignment

Each goal must be a SMART, measurable statement.
For the first three: "[Condition], [Student] will [behavior] [criteria] [timeframe]."
For 'benchmarks' and 'alignment': arrays of short strings.

Profile JSON:
{json.dumps(profile_json, indent=2)}
"""

def build_rag_prompt(profile_json: dict, docs: list[dict]) -> str:
    # build concise context
    lines = []
    for d in docs:
        snippet = d['text'].replace('\n',' ')[:300]
        label = "Occupational Info" if d['source']=='bls_ooh' else "Standards"
        lines.append(f"{label}: {snippet}… [SOURCE:{d['section']}]")
    context = "\n\n---\n".join(lines)

    question = generate_goals_prompt(profile_json)
    return f"""You are an educational planning assistant. Use ONLY the CONTEXT below.

CONTEXT:
{context}

QUESTION:
{question}
"""

# ── 5) Zero-login LLM setup (Flan-T5) ───────────────────────────────────────────
LLM_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(
    LLM_NAME,
    device_map="auto",
    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
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

# ── 6) LLM wrapper ─────────────────────────────────────────────────────────────
def llm(prompt: str) -> dict:
    print("LLM PROMPT:\n", prompt)
    out = generator(prompt)[0]["generated_text"]
    print("LLM RAW OUTPUT:\n", out)
    try:
        s = out.find("{"); e = out.rfind("}") + 1
        return json.loads(out[s:e])
    except Exception:
        return {"raw_output": out.strip()}

# ── 7) End-to-end pipeline ─────────────────────────────────────────────────────
def process_student_profile(profile_text: str) -> dict:
    # Extract structured JSON
    profile_json = llm(extract_student_info_prompt(profile_text))
    # Retrieve only OOH + standards
    docs = retrieve_context(profile_json.get("postsecondary_goal_employment","undecided"))
    # Build prompt and generate
    prompt = build_rag_prompt(profile_json, docs)
    return llm(prompt)
    return llm(prompt)
