# ✅ FINAL_NLP_Course_CLEAN.py — Cleaned up for OpenAI integration and modular RAG logic

import requests
from bs4 import BeautifulSoup
import pdfplumber
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import openai
import json

# ✅ Load OpenAI key from environment only
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ FAISS config
INDEX_PATH     = os.path.join('data', 'full_rag.index')
META_PATH      = os.path.join('data', 'full_rag_metadata.pkl')
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVE_K     = 3

# ✅ Load FAISS index and metadata
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta    = pickle.load(open(META_PATH, "rb"))

# ✅ Retrieve top-K matching documents

def retrieve(query: str, k: int = RETRIEVE_K, embedder=None):
    embedder = embedder or SentenceTransformer(EMBED_MODEL)
    q_vec = embedder.encode(query, normalize_embeddings=True)
    D, I  = faiss_index.search(q_vec.reshape(1, -1).astype("float32"), k)
    return [dict(doc_meta[i], sim=float(score)) for score, i in zip(D[0], I[0])]

# ✅ Prompt builders
SYSTEM_PROMPT = (
    "You are an educational planning assistant. "
    "Use only the CONTEXT provided. Cite facts as [SOURCE:<section>]."
)

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
    return f"""Based on the student's profile and postsecondary employment and education goals,
write three SMART annual IEP goals — one for each of the following areas: academic achievement,
independent living skills, and career preparation. Goals must follow this structure:

[Condition], [Student] will [behavior] [criteria] [timeframe].

Use the following profile:
{structured_profile_json}
"""

def make_prompt(question: str, docs: list[dict]) -> str:
    context = "\n\n---\n".join(
        f"{doc['text']} [SOURCE:{doc['section']}]" for doc in docs
    )
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n<|assistant|>\n"

# ✅ OpenAI-based LLM function

def llm(prompt: str) -> dict:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that writes SMART IEP goals based on a student profile and career standards. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    output = response["choices"][0]["message"]["content"]
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"raw_output": output}

# ✅ Processing pipeline

def process_student_profile(profile_text: str):
    extraction_prompt = extract_student_info_prompt(profile_text)
    structured_info = llm(extraction_prompt)
    goal_query = structured_info.get("postsecondary goal (employment)", "undecided")
    docs = retrieve(goal_query)
    question = generate_goals_prompt(structured_info)
    full_prompt = make_prompt(question, docs)
    return llm(full_prompt)

