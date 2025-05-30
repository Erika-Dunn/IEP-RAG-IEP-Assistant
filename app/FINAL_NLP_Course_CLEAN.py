# FINAL_NLP_Course_CLEAN.py

import os
import json
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ─── 1) Load FAISS index & metadata ─────────────────────────────────────────────
INDEX_PATH = os.path.join('data', 'full_rag.index')
META_PATH  = os.path.join('data', 'full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta     = pickle.load(open(META_PATH, 'rb'))

# ─── 2) Embedding model ─────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def vector_search(query: str, k: int = 3) -> list:
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(vec.astype("float32"), k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# ─── 3) Filtered RAG retrieval ───────────────────────────────────────────────────
def retrieve_context(goal_query: str) -> list:
    # a) Career context (BLS OOH only)
    occ = [h for h in vector_search(goal_query, k=3) if h['source']=="bls_ooh"]
    # b) Standards context (21st Century Skills only)
    std = [h for h in vector_search("21st Century Skills "+goal_query, k=2)
           if h['source'].lower().startswith("or-21st")]
    # merge & dedupe by section
    merged = {h['section']: h for h in (occ + std)}
    return list(merged.values())

# ─── 4) Prompt builders ──────────────────────────────────────────────────────────
def extract_student_info_prompt(profile_text: str) -> str:
    return (
        "Extract JSON with keys "
        "[name, age, grade_level, disability, strengths, academic_concerns, "
        "support_needs, postsecondary_goal_employment, "
        "postsecondary_goal_education, postsecondary_goal_independent_living]\n\n"
        f"Profile:\n\"\"\"{profile_text}\"\"\""
    )

def build_rag_prompt(profile_json: dict, docs: list) -> str:
    # Two pure JSON examples—no labels, no commentary
    examples = [
      {
        "employment_goal": "After high school, Clarence will obtain a full-time job at Walmart as a sales associate.",
        "education_goal": "After high school, Clarence will complete on-the-job training and customer service workshops.",
        "annual_goal": "In 36 weeks, Clarence will demonstrate effective customer service skills in 4 out of 5 role-plays.",
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
      },
      {
        "employment_goal": "After graduation, Marisol will secure a veterinary assistant position.",
        "education_goal": "Within 6 months, Marisol will complete an accredited animal care certificate program.",
        "annual_goal": "By year-end, Marisol will demonstrate proper animal handling in 3 supervised labs.",
        "benchmarks": [
          "Identify basic animal anatomy",
          "Perform feeding tasks",
          "Follow safety protocols"
        ],
        "alignment": [
          "OOH standards for Veterinary Assistants",
          "State Science Standards"
        ]
      }
    ]
    # join examples with blank line
    example_str = "\n\n".join(json.dumps(e) for e in examples)

    # build concise context
    lines = []
    for d in docs:
        label = "Career" if d['source']=="bls_ooh" else "Standard"
        snippet = d['text'].replace("\n"," ")[:200]
        lines.append(f"{label}: {snippet}… [SRC:{d['section']}]")
    context = "\n\n---\n".join(lines)

    return (
        example_str
        + "\n\n"
        + "#### Now, using ONLY the STUDENT PROFILE and CONTEXT below, "
          "return EXACTLY a SINGLE JSON object with these keys (no extra text!):\n"
        "- employment_goal\n"
        "- education_goal\n"
        "- annual_goal\n"
        "- benchmarks\n"
        "- alignment\n\n"
        f"STUDENT PROFILE:\n{json.dumps(profile_json)}\n\n"
        f"CONTEXT:\n{context}"
    )

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
def llm_call(prompt: str) -> dict:
    print("PROMPT:\n", prompt)
    out = generator(prompt)[0]["generated_text"]
    print("RAW OUTPUT:\n", out)
    try:
        start = out.find("{")
        end   = out.rfind("}") + 1
        return json.loads(out[start:end])
    except:
        return {"raw_output": out.strip()}

# ─── 7) End-to-end pipeline ─────────────────────────────────────────────────────
def process_student_profile(profile_text: str) -> dict:
    # 1) Extract structured JSON profile
    info_json = llm_call(extract_student_info_prompt(profile_text))
    # 2) Retrieve only Career + Standards
    docs = retrieve_context(info_json.get("postsecondary_goal_employment","undecided"))
    # 3) Build RAG prompt & generate
    rag_prompt = build_rag_prompt(info_json, docs)
    return llm_call(rag_prompt)
    return llm(prompt)
