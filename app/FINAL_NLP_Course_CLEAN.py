# FINAL_NLP_Course_CLEAN.py

import os, json, pickle, faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ─── 1) Load FAISS index & metadata ─────────────────────────────────────────────
INDEX_PATH = os.path.join('data','full_rag.index')
META_PATH  = os.path.join('data','full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta     = pickle.load(open(META_PATH,'rb'))

# ─── 2) Embedder for retrieval ──────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
def vector_search(query:str, k:int=3)->list:
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(vec.astype("float32"), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j,i in enumerate(I[0])]

# ─── 3) Filtered RAG retrieval ───────────────────────────────────────────────────
def retrieve_context(goal_query:str)->list:
    occ = [h for h in vector_search(goal_query,3) if h['source']=='bls_ooh']
    std = [h for h in vector_search("21st Century Skills "+goal_query,2)
           if h['source'].lower().startswith('or-21st')]
    merged = {h['section']:h for h in (occ+std)}
    return list(merged.values())

# ─── 4) Prompt builders ──────────────────────────────────────────────────────────
def extract_student_info_prompt(profile_text:str)->str:
    return (
        "Extract JSON with these keys: "
        "[name, age, grade_level, disability, strengths, academic_concerns, "
        "support_needs, postsecondary_goal_employment, postsecondary_goal_education, "
        "postsecondary_goal_independent_living]\n\n"
        f"Profile:\n\"\"\"{profile_text}\"\"\""
    )

def build_rag_prompt(profile_json:dict, docs:list)->str:
    # Build a concise context block
    lines=[]
    for d in docs:
        label = "Career" if d['source']=='bls_ooh' else "Standard"
        snippet = d['text'].replace("\n"," ")[:200]
        lines.append(f"{label}: {snippet}… [SRC:{d['section']}]")
    context="\n\n---\n".join(lines)

    # Wrap desired response in clear tags
    return f"""
You are an educational planning assistant. Use ONLY the CONTEXT below.

CONTEXT:
{context}

STUDENT PROFILE JSON:
{json.dumps(profile_json)}

Now **return ONLY** the JSON between the `<JSON>` and `</JSON>` tags, with exactly these keys:
- employment_goal
- education_goal
- annual_goal
- benchmarks
- alignment

No extra text outside the tags.

<JSON>
{{}}
</JSON>

(Note: Replace the empty braces above with your JSON object.)
"""

# ─── 5) Zero-login LLM setup ─────────────────────────────────────────────────────
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

# ─── 6) LLM wrapper ─────────────────────────────────────────────────────────────
def llm_call(prompt:str)->dict:
    print("PROMPT:\n",prompt)
    out = generator(prompt)[0]["generated_text"]
    print("RAW OUTPUT:\n",out)
    # Extract JSON between tags
    start = out.find("<JSON>")
    end   = out.find("</JSON>")
    if start!=-1 and end!=-1:
        json_str = out[start+6:end].strip()
        try:
            return json.loads(json_str)
        except:
            pass
    # fallback: try to find braces
    try:
        s,e = out.find("{"), out.rfind("}")+1
        return json.loads(out[s:e])
    except:
        return {"raw_output":out.strip()}

# ─── 7) End-to-end pipeline ─────────────────────────────────────────────────────
def process_student_profile(profile_text:str)->dict:
    # A) Extract structured JSON
    info = llm_call(extract_student_info_prompt(profile_text))
    # B) Retrieve context
    docs = retrieve_context(info.get("postsecondary_goal_employment","undecided"))
    # C) Build RAG prompt & generate final goals
    prompt = build_rag_prompt(info, docs)
    return llm_call(prompt)
