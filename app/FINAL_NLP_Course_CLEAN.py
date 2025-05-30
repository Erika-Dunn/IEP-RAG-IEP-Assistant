# FINAL_NLP_Course_CLEAN.py
import os
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# 1) Load FAISS index and metadata
INDEX_PATH = os.path.join('data', 'full_rag.index')
META_PATH  = os.path.join('data', 'full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta     = pickle.load(open(META_PATH, 'rb'))

# 2) Embedding model for retrieval
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def vector_search(query: str, k: int = 3) -> list:
    q_vec = embedder.encode([query], normalize_embeddings=True)
    D, I  = faiss_index.search(q_vec.astype('float32'), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# 3) Aggregate retrieval across data sources
def retrieve_context(goal_query: str) -> list:
    occ_hits = vector_search(goal_query, k=3)
    std_hits = vector_search("21st Century Skills " + goal_query, k=2)
    iep_hits = vector_search("IEP goal example", k=1)
    # merge and dedupe by section
    merged = {h['section']: h for h in (occ_hits + std_hits + iep_hits)}
    return list(merged.values())

# 4) Prompt templates
def extract_student_info_prompt(profile_text: str) -> str:
    return f"""Extract the following structured information from the student's profile text below.
If information is not present, write "missing". Output in JSON format with these keys:
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
\"\"\"{profile_text}\"\"\""""

def generate_goals_prompt(profile_json: dict) -> str:
    return f"""Based on the student profile below (in JSON), return ONLY a JSON object with exactly these five keys:
- postsecondary_employment_goal
- postsecondary_education_goal
- annual_goal
- benchmarks
- alignment

Each goal must be SMART and measurable.  
For the first three keys: a single goal string in the format  
[Condition], [Student] will [behavior] [criteria] [timeframe].  
For 'benchmarks' and 'alignment': arrays of short strings.

Student profile JSON:
{json.dumps(profile_json, indent=2)}
"""

def build_rag_prompt(profile_json: dict, docs: list) -> str:
    # Label and truncate each context doc
    lines = []
    for d in docs:
        src = d['source']
        text = d['text'].replace('\n',' ')[:300]
        if src == 'bls_ooh':
            label = d.get('job_title','Occupational Info')
        elif src.startswith('or_21stCentury'):
            label = 'Standards'
        else:
            label = 'IEP Example'
        lines.append(f"{label}: {text}… [SOURCE:{d['section']}]")
    context = "\n\n---\n".join(lines)

    question = generate_goals_prompt(profile_json)
    return f"""You are an educational planning assistant. Use ONLY the CONTEXT below (do NOT repeat it).
CONTEXT:
{context}

QUESTION:
{question}
"""

# 5) Load an instruction-tuned model
LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model     = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    temperature=0.0,
    return_full_text=False,
)

# 6) LLM wrapper that tries JSON parse, else raw
def llm(prompt: str) -> dict:
    print("LLM PROMPT:\n", prompt)
    out = generator(prompt)[0]["generated_text"]
    print("LLM RAW OUTPUT:\n", out)
    try:
        start = out.find("{")
        end   = out.rfind("}") + 1
        return json.loads(out[start:end])
    except Exception:
        return {"raw_output": out.strip()}

# 7) End-to-end pipeline
def process_student_profile(profile_text: str) -> dict:
    # 1) Extract structured student info
    info = llm(extract_student_info_prompt(profile_text))

    # 2) Retrieve context
    docs = retrieve_context(info.get("postsecondary_goal_employment","undecided"))

    # 3) Build RAG prompt and generate
    prompt = build_rag_prompt(info, docs)
    return llm(prompt)
    return llm(full_prompt)
