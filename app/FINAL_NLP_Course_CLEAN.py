# FINAL_NLP_Course_CLEAN.py

import os, pickle, torch, faiss, re, json
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ─── Configuration ────────────────────────────────────────────────────────────
INDEX_PATH     = "data/full_rag.index"
META_PATH      = "data/full_rag_metadata.pkl"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME       = "HuggingFaceH4/zephyr-7b-beta"
RETRIEVE_K     = 3
MAX_NEW_TOKENS = 300
DEVICE         = 0 if torch.cuda.is_available() else "cpu"

# ─── Load FAISS Index and Metadata ────────────────────────────────────────────
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta    = pickle.load(open(META_PATH, "rb"))

# ─── Embedding and Retrieval ──────────────────────────────────────────────────
def retrieve(query: str, k: int = RETRIEVE_K, embedder=None):
    embedder = embedder or SentenceTransformer(EMBED_MODEL)
    q_vec = embedder.encode(query, normalize_embeddings=True)
    D, I  = faiss_index.search(q_vec.reshape(1, -1).astype("float32"), k)
    return [dict(doc_meta[i], sim=float(score)) for score, i in zip(D[0], I[0])]

# ─── Prompting ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an educational planning assistant. "
    "Use only the CONTEXT provided. Cite facts as [SOURCE:<section>]."
)

def make_prompt(question: str, docs: List[dict]) -> str:
    context = "\n\n---\n".join(f"{doc['text']} [SOURCE:{doc['section']}]" for doc in docs)
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n<|assistant|>\n"

# ─── LLM Generation ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=True, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS
)

def rag_chat(question: str, k: int = RETRIEVE_K, max_new_tokens: int = MAX_NEW_TOKENS):
    hits = retrieve(question, k=k)
    prompt = make_prompt(question, hits)

    encodings = tokenizer(prompt, return_tensors="pt", truncation=True,
                          padding=True, max_length=8192 - max_new_tokens)
    input_ids = encodings.input_ids.to(generator.model.device)
    attention_mask = encodings.attention_mask.to(generator.model.device)

    output_ids = generator.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = output.split("<|assistant|>")[-1].strip()
    return answer, hits

# ─── Prompt Engineering ───────────────────────────────────────────────────────
def extract_student_info_prompt(profile_text: str) -> str:
    return f'''
    Extract the following structured information from the student's profile text below.
    If information is not present, write "missing". Output in JSON format with these fields:
    - name
    - age
    - grade level
    - disability
    - strengths
    - academic concerns
    - support needs
    - postsecondary goal (employment)
    - postsecondary goal (education)
    - postsecondary goal (independent living)

    Student profile:
    """{profile_text}"""
    '''

def generate_goals_prompt(structured_profile_json: dict) -> str:
    return f"""Based on the student's profile and postsecondary employment and education goals,
write three SMART annual IEP goals — one for each of the following areas: academic achievement,
independent living skills, and career preparation. Goals must follow this structure:

[Condition], [Student] will [behavior] [criteria] [timeframe].

Use the following profile:
{json.dumps(structured_profile_json, indent=2)}
"""

# ─── Pipeline Wrapper ─────────────────────────────────────────────────────────
def process_student_profile(profile_text: str) -> dict:
    extraction_prompt = extract_student_info_prompt(profile_text)
    extracted_json_str, _ = rag_chat(extraction_prompt)
    structured_info = json.loads(extracted_json_str)

    goal_query = structured_info.get("postsecondary goal (employment)", "undecided")
    docs = retrieve(goal_query)

    question = generate_goals_prompt(structured_info)
    final_output, _ = rag_chat(question)

    return json.loads(final_output)
