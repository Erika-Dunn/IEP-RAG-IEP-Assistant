# FINAL_NLP_Course_CLEAN.py

import os, json, pickle, faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 1) Load FAISS index & metadata
INDEX_PATH = os.path.join('data','full_rag.index')
META_PATH  = os.path.join('data','full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta    = pickle.load(open(META_PATH,'rb'))

# 2) Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
def vector_search(query, k=3):
    vec = embedder.encode([query], normalize_embeddings=True)
    D,I = faiss_index.search(vec.astype("float32"), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j,i in enumerate(I[0])]

# 3) Filtered RAG retrieval
def retrieve_context(goal_q):
    # BLS OOH only
    occ = [h for h in vector_search(goal_q,3) if h['source']=="bls_ooh"]
    # 21stC only
    std = [h for h in vector_search("21st Century Skills "+goal_q,2)
           if h['source'].lower().startswith("or-21st")]
    # dedupe
    uniq = {h['section']:h for h in occ+std}
    return list(uniq.values())

# 4) Prompt builders
def extract_student_info_prompt(profile_text):
    return (f"Extract JSON with keys name, age, grade_level, disability, strengths, "
            f"academic_concerns, support_needs, postsecondary_goal_employment, "
            f"postsecondary_goal_education, postsecondary_goal_independent_living\n\n"
            f"Profile:\n\"\"\"{profile_text}\"\"\"")

def build_rag_prompt(profile_json, docs):
    # build context
    lines=[]
    for d in docs:
        lbl = "Career" if d['source']=="bls_ooh" else "Standard"
        snippet = d['text'].replace("\n"," ")[:200]
        lines.append(f"{lbl}: {snippet}… [SRC:{d['section']}]")
    context = "\n\n---\n".join(lines)

    # precise JSON task
    return (
        "You are an educational planning assistant. Use ONLY the CONTEXT below.\n\n"
        f"CONTEXT:\n{context}\n\n"
        "STUDENT PROFILE (JSON):\n"
        f"{json.dumps(profile_json,indent=2)}\n\n"
        "RETURN EXACTLY and ONLY a JSON object with these fields:\n"
        "  employment_goal\n"
        "  education_goal\n"
        "  annual_goal\n"
        "  benchmarks\n"
        "  alignment\n"
        "No extra text.\n"
    )

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
    "text2text-generation", model=model, tokenizer=tokenizer,
    max_new_tokens=200, do_sample=False, temperature=0.0, return_full_text=False
)

# 6) LLM wrapper
def llm_call(prompt:str) -> dict:
    print("PROMPT:\n",prompt)
    out = generator(prompt)[0]["generated_text"]
    print("RAW OUTPUT:\n",out)
    try:
        s,e = out.find("{"), out.rfind("}")+1
        return json.loads(out[s:e])
    except:
        return {"raw_output": out.strip()}

# 7) End-to-end pipeline
def process_student_profile(profile_text:str) -> dict:
    info = llm_call(extract_student_info_prompt(profile_text))
    docs = retrieve_context(info.get("postsecondary_goal_employment","undecided"))
    prompt = build_rag_prompt(info, docs)
    return llm_call(prompt)
    return llm(prompt)
