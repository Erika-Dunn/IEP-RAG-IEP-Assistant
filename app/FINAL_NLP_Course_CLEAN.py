# FINAL_NLP_Course_CLEAN.py

import os
import json
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# 1) Load FAISS index & metadata
INDEX_PATH = os.path.join('data', 'full_rag.index')
META_PATH  = os.path.join('data', 'full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta     = pickle.load(open(META_PATH, 'rb'))

# 2) Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def vector_search(query: str, k: int = 3) -> list:
    q_vec = embedder.encode([query], normalize_embeddings=True)
    D, I  = faiss_index.search(q_vec.astype('float32'), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# 3) RAG retrieval (only BLS OOH + 21st Century Skills)
def retrieve_context(goal_query: str) -> list:
    # Occupational context
    occ_hits = [h for h in vector_search(goal_query, k=3) if h['source']=='bls_ooh']
    # Standards context
    std_hits = [h for h in vector_search("21st Century Skills " + goal_query, k=2)
                if h['source'].lower().startswith('or-21st')]
    merged = {h['section']: h for h in (occ_hits + std_hits)}
    return list(merged.values())

# 4) Prompt templates
def extract_student_info_prompt(profile_text: str) -> str:
    return f"""Extract the following structured information from the student's profile below.
If not present, write "missing". Output JSON with keys:
- name
- age
- grade_level
- disability
- strengths
- academic_concerns
- support_needs
- postsecondary_goal_employment
- postsecondary_goal_education
- postsecondary_goal_independent_living

Student profile:
\"\"\"{profile_text}\"\"\"
"""

def generate_goals_prompt(profile_json: dict) -> str:
    return f"""Based on the student profile below (in JSON), return **only** a JSON object with exactly these five keys:
1. postsecondary_employment_goal  
2. postsecondary_education_goal  
3. annual_goal  
4. benchmarks  
5. alignment  

Each of the first three keys must be a SMART goal string:
[Condition], [Student] will [behavior] [criteria] [timeframe].  
For 'benchmarks' and 'alignment', return arrays of short strings.

Student profile JSON:
{json.dumps(profile_json, indent=2)}
"""

def build_rag_prompt(profile_json: dict, docs: list) -> str:
    # Label & truncate
    lines = []
    for d in docs:
        text = d['text'].replace('\n',' ')[:300]
        label = "Occupational Info" if d['source']=='bls_ooh' else "Standards"
        lines.append(f"{label}: {text}… [SOURCE:{d['section']}]")
    context = "\n\n---\n".join(lines)

    question = generate_goals_prompt(profile_json)
    return f"""You are an educational planning assistant. Use only the CONTEXT below (do NOT repeat it).

CONTEXT:
{context}

QUESTION:
{question}
"""

# 5) Zero-login LLM (Flan-T5 Base)
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

# 6) LLM wrapper
def llm(prompt: str) -> dict:
    print("LLM PROMPT:\n", prompt)
    out = generator(prompt)[0]["generated_text"]
    print("LLM RAW OUTPUT:\n", out)
    try:
        s = out.find("{")
        e = out.rfind("}") + 1
        return json.loads(out[s:e])
    except:
        return {"raw_output": out.strip()}

# 7) End-to-end
def process_student_profile(profile_text: str) -> dict:
    # Extract structured info
    info = llm(extract_student_info_prompt(profile_text))
    # Retrieve context
    docs = retrieve_context(info.get("postsecondary_goal_employment","undecided"))
    # Build & send prompt
    prompt = build_rag_prompt(info, docs)
    return llm(prompt)
