import os, json, pickle, faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ─── 1) Load & index your data ─────────────────────────────────────────────────
INDEX_PATH = "data/full_rag.index"
META_PATH  = "data/full_rag_metadata.pkl"
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta     = pickle.load(open(META_PATH, "rb"))

# ─── 2) Embedding model ─────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
def vector_search(q, k=3):
    vec = embedder.encode([q], normalize_embeddings=True)
    D, I  = faiss_index.search(vec.astype("float32"), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j,i in enumerate(I[0])]

# ─── 3) Filtered retrieval ───────────────────────────────────────────────────────
def retrieve_context(goal_q):
    # a) Career context (only bls_ooh)
    occ = [h for h in vector_search(goal_q, k=3) if h["source"]=="bls_ooh"]
    # b) Standards context (only 21st Century)
    std = [h for h in vector_search("21st Century Skills "+goal_q, k=2)
           if h["source"].lower().startswith("or-21stcentury")]
    # merge & dedupe
    uniq = {h["section"]: h for h in occ+std}
    return list(uniq.values())

# ─── 4) Prompt builder ──────────────────────────────────────────────────────────
def build_prompt(profile_json, docs):
    # few-shot examples
    ex1 = {
      "employment_goal": "After high school, Clarence will obtain a full-time job at Walmart as a sales associate.",
      "education_goal": "After high school, Clarence will complete on-the-job training and customer service workshops.",
      "annual_goal": "In 36 weeks, Clarence will demonstrate effective customer service skills in 4/5 role-plays.",
      "benchmarks": ["Greet customers","Maintain eye contact","Listen actively","Respond to questions"],
      "alignment": ["OOH Retail Sales","21st Century Skills"]
    }
    ex2 = {
      "employment_goal": "After graduation, Linh will secure a junior graphic designer internship.",
      "education_goal": "Within 1 year, Linh will complete an introductory design course.",
      "annual_goal": "By year-end, Linh will design 3 digital mockups with 80% accuracy.",
      "benchmarks": ["Sketch basic shapes","Use design software","Incorporate feedback"],
      "alignment": ["OOH Graphic Designers","State Art Standards"]
    }
    # build context
    ctx = []
    for d in docs:
        label = "Career" if d["source"]=="bls_ooh" else "Standard"
        txt   = d["text"].replace("\n"," ")[:300]
        ctx.append(f"{label}: {txt}… [SRC:{d['section']}]")
    context = "\n\n---\n".join(ctx)

    return f"""Here are two examples of the **exact** JSON format (no code, no extra text):

Example 1:
{json.dumps(ex1, indent=2)}

Example 2:
{json.dumps(ex2, indent=2)}

Using ONLY the CONTEXT below—and the STUDENT PROFILE—return **only** a JSON object with these keys:
- employment_goal
- education_goal
- annual_goal
- benchmarks
- alignment

CONTEXT:
{context}

STUDENT PROFILE:
{json.dumps(profile_json, indent=2)}
"""

# ─── 5) Zero-login LLM setup ─────────────────────────────────────────────────────
LLM="google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(LLM)
model     = AutoModelForSeq2SeqLM.from_pretrained(
    LLM, device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
generator = pipeline(
    "text2text-generation", model=model, tokenizer=tokenizer,
    max_new_tokens=200, do_sample=False, temperature=0.0,
    return_full_text=False
)

# ─── 6) LLM wrapper ─────────────────────────────────────────────────────────────
def llm_call(prompt:str)->dict:
    print("PROMPT:\n",prompt)
    out = generator(prompt)[0]["generated_text"]
    print("RAW OUT:\n",out)
    try:
        s,e = out.find("{"), out.rfind("}")+1
        return json.loads(out[s:e])
    except:
        return {"raw_output":out.strip()}

# ─── 7) End-to-end pipeline ─────────────────────────────────────────────────────
def process_student_profile(profile_text:str)->dict:
    # A) turn text into JSON profile
    info_prompt = f"""Extract JSON from the profile below with keys:
name, age, grade_level, disability, strengths,
academic_concerns, support_needs,
postsecondary_goal_employment,
postsecondary_goal_education,
postsecondary_goal_independent_living

\"\"\"{profile_text}\"\"\""""
    profile_json = llm_call(info_prompt)

    # B) retrieve career+standard context
    docs = retrieve_context(profile_json.get("postsecondary_goal_employment","undecided"))

    # C) build prompt & generate goals
    prompt = build_prompt(profile_json, docs)
    return llm_call(prompt)
    return llm(prompt)
