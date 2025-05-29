# FINAL_NLP_Course_CLEAN.py
# ✅ Core logic: RAG pipeline, OpenAI LLM call, student profile processing

import os
import json
import pickle
import re
import faiss
import openai
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index and metadata
INDEX_PATH = os.path.join('..', 'data', 'full_rag.index')
META_PATH = os.path.join('..', 'data', 'full_rag_metadata.pkl')
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta = pickle.load(open(META_PATH, "rb"))

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
    return f"""Based on the student's profile and postsecondary employment and education goals,
write three SMART annual IEP goals — one for each of the following areas: academic achievement,
independent living skills, and career preparation. Goals must follow this structure:

[Condition], [Student] will [behavior] [criteria] [timeframe].

Use the following profile:
{structured_profile_json}
"""

def make_prompt(question: str, docs: list[dict]) -> str:
    context = "\n\n---\n".join(f"{doc['text']} [SOURCE:{doc['section']}]" for doc in docs)
    return f"""
You are an educational planning assistant. Use only the CONTEXT provided. Cite facts as [SOURCE:<section>].

CONTEXT:
{context}

QUESTION:
{question}
"""

# --- OpenAI LLM ---
def llm(prompt: str) -> dict:
    print("LLM PROMPT:\n", prompt)
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
        print("⚠️ Warning: LLM did not return valid JSON. Returning raw output.")
        return {"raw_output": output}

# --- Main Pipeline ---
def process_student_profile(profile_text: str) -> dict:
    extraction_prompt = extract_student_info_prompt(profile_text)
    structured_info = llm(extraction_prompt)

    goal_query = structured_info.get("postsecondary goal (employment)", "undecided")
    docs = vector_search(goal_query)

    question = generate_goals_prompt(structured_info)
    full_prompt = make_prompt(question, docs)

    return llm(full_prompt)


