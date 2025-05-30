# FINAL_NLP_Course_CLEAN.py
# ✅ Core logic: RAG pipeline, Hugging Face LLM call, student profile processing

import os
import json
import pickle
import re
import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load FAISS index and metadata
INDEX_PATH = os.path.join('data', 'full_rag.index')
META_PATH = os.path.join('data', 'full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta = pickle.load(open(META_PATH, "rb"))

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Replace existing LLM_NAME / MAX_NEW_TOKENS lines with:
LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW_TOKENS = 200

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# ↓ Delete or comment out your old generator definition, then paste this ↓
generator = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ),
    tokenizer=AutoTokenizer.from_pretrained(LLM_NAME),
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,           # deterministic decoding
    temperature=0.0,           # follow instructions exactly
    return_full_text=False     # don't echo the prompt back
)

# --- Vector Search ---
def vector_search(query: str, k: int = 3) -> list:
    q_vec = embedder.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(q_vec, k)
    return [dict(doc_meta[i], sim=float(D[0][j])) for j, i in enumerate(I[0])]

# --- Prompt Engineering ---
def extract_student_info_prompt(profile_text):
    return f"""Extract the following structured information from the student's profile text below.
If information is not present, write \"missing\". Output in JSON format with these fields:
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
\"\"\"{profile_text}\"\"\"
"""
    
def generate_goals_prompt(structured_profile_json):
    return f"""You are an educational planning assistant.  

**Return ONLY** a JSON object with exactly three keys (no extra text, no code):

  • academic_goal  
  • independent_living_goal  
  • career_preparation_goal  

Each value must be a SMART IEP goal in this format:  
[Condition], [Student] will [behavior] [criteria] [timeframe].

Student profile:
{structured_profile_json}
"""
    
def make_prompt(question: str, docs: list[dict]) -> str:
    # only keep the first 500 chars of each doc to avoid overload
    context = "\n\n---\n".join(
        f"{doc['text'][:500]} [SOURCE:{doc['section']}]"
        for doc in docs
    )
    return f"""
You are an educational planning assistant.  
**Use ONLY the CONTEXT below**—do NOT repeat it in your answer.  
Answer in JSON with keys: academic_goal, independent_living_goal, career_preparation_goal.

CONTEXT:
{context}

QUESTION:
{question}
"""

# --- Local LLM ---
def llm(prompt: str) -> dict:
    print("LLM PROMPT:\n", prompt)
    response = generator(prompt)[0]["generated_text"]
    print("LLM RAW OUTPUT:\n", response)

    # If the model used your prompt back, only keep the actual answer 
    if "Condition:" in response:
        # Chop away everything up to the first “Condition:”
        answer = "Condition:" + response.split("Condition:", 1)[1].strip()
        return {"raw_output": answer}

    # Otherwise just return whatever it gave you
    return {"raw_output": response.strip()}

# --- Main Pipeline ---
def process_student_profile(profile_text: str) -> dict:
    # … extraction prompt, llm, etc …

    goal_query = structured_info.get("postsecondary goal (employment)", "undecided")
    docs = vector_search(goal_query)

    # 🔥 Keep only the BLS OOH sections (source == 'bls_ooh')
    ooh_docs = [d for d in docs if d["source"] == "bls_ooh"]
    # fallback to original docs if no OOH found
    docs = ooh_docs or docs

    question = generate_goals_prompt(structured_info)
    full_prompt = make_prompt(question, docs)
    return llm(full_prompt)
