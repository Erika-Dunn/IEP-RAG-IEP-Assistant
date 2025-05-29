#!/usr/bin/env python
# coding: utf-8

# # Section 0: Prepping for GitHub

# In[ ]:


# Set Git identity

# Clone your GitHub repo

# Change directory


# 
# 
# # Section 1: Data Collection and Preprocessing

# In[5]:


# Install Required Packages



# In[6]:


# Import Libraries
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


# In[8]:


# Google Drive


# In[ ]:


# Load BLS OOH XML and chunk
file_path = '/content/drive/MyDrive/Colab Notebooks/NLP Course/ooh_occupations.xml'
tree = ET.parse(file_path)
root = tree.getroot()

def extract_text(el):
    return ''.join(el.itertext()).strip() if el is not None else None

data = []
for occ in root.findall('.//occupation'):
    data.append({
        'job_code': extract_text(occ.find('occupation_code')),
        'job_title': extract_text(occ.find('title')),
        'what_they_do': extract_text(occ.find('what_they_do') or occ.find('summary_what_they_do')),
        'work_environment': extract_text(occ.find('work_environment') or occ.find('summary_work_environment')),
        'how_to_become': extract_text(occ.find('how_to_become_one') or occ.find('summary_how_to_become_one')),
        'pay': extract_text(occ.find('pay') or occ.find('summary_pay')),
        'job_outlook': extract_text(occ.find('job_outlook') or occ.find('summary_outlook')),
        'similar_occupations': extract_text(occ.find('similar_occupations') or occ.find('summary_similar_occupations'))
    })

ooh_df = pd.DataFrame(data)
ooh_chunks = []
for _, row in ooh_df.iterrows():
    for sec in ['what_they_do', 'work_environment', 'how_to_become', 'pay', 'job_outlook', 'similar_occupations']:
        text = row[sec]
        if pd.notna(text) and text.strip():
            ooh_chunks.append({
                'source': 'bls_ooh',
                'section': sec,
                'text': re.sub(r'\s+', ' ', text.strip()),
                'job_code': row['job_code'],
                'job_title': row['job_title']
            })

# Oregon IEP - Web and PDF
url = "https://www.oregon.gov/ode/students-and-family/SpecialEducation/publications/Pages/Oregon-Standard-IEP.aspx"
soup = BeautifulSoup(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text, "html.parser")
web_lines = [line.strip() for line in soup.get_text("\n").splitlines() if len(line.strip()) > 30]
web_chunks = [{"source": "oregon_iep_web", "section": f"paragraph_{i+1}", "text": line, "job_code": None, "job_title": None}
              for i, line in enumerate(web_lines) if len(line.split()) > 10]

pdf_links = [a['href'] if a['href'].startswith('http') else f"https://www.oregon.gov{a['href']}"
             for a in soup.find_all("a", href=True) if '.pdf' in a['href'].lower()]

os.makedirs("oregon_iep_pdfs", exist_ok=True)
pdf_chunks = []
for i, link in enumerate(pdf_links):
    filename = f"oregon_iep_pdfs/doc_{i+1}.pdf"
    r = requests.get(link)
    with open(filename, "wb") as f:
        f.write(r.content)
    with pdfplumber.open(filename) as pdf:
        for j, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            for k, chunk in enumerate(txt.split("\n\n")):
                if len(chunk.split()) > 10:
                    pdf_chunks.append({
                        "source": "oregon_iep_pdf",
                        "section": f"doc_{i+1}_pg{j+1}_blk{k+1}",
                        "text": chunk.strip(),
                        "job_code": None,
                        "job_title": None
                    })

# NASET IEP PDF Chunking
naset_path = "/content/drive/MyDrive/Colab Notebooks/NLP Course/Completed_Sample_IEP.pdf"
with pdfplumber.open(naset_path) as pdf:
    naset_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

naset_chunks = []
raw_chunks = re.split(r'(Measurable Postsecondary Goal.*?:|Annual Goal.*?:|Short[- ]Term Objective.*?:)', naset_text)
for i in range(1, len(raw_chunks), 2):
    label = raw_chunks[i].strip()
    content = raw_chunks[i+1].strip()
    if len(content.split()) >= 10:
        naset_chunks.append({
            "source": "naset_iep_example",
            "section": label,
            "text": content,
            "job_code": None,
            "job_title": None
        })

# Merge all chunks into one DF
combined_df = pd.DataFrame(ooh_chunks + web_chunks + pdf_chunks + naset_chunks)

# Embedding + FAISS
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = combined_df['text'].tolist()
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "/content/drive/MyDrive/Colab Notebooks/NLP Course/full_rag.index")
with open("/content/drive/MyDrive/Colab Notebooks/NLP Course/full_rag_metadata.pkl", "wb") as f:
    pickle.dump(combined_df.to_dict(orient='records'), f)

print(f"✅ Combined RAG index created with {len(combined_df)} records.")


# # Section 2: RAG Pipeline Implementation

# # Section 2 · RAG Retrieval + Generation (Zephyr 7B, FAISS, Colab Pro)

# In[ ]:




# In[ ]:


import os, pickle, torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# In[ ]:


# Config
INDEX_PATH     = "/content/drive/MyDrive/Colab Notebooks/NLP Course/full_rag.index"
META_PATH      = "/content/drive/MyDrive/Colab Notebooks/NLP Course/full_rag_metadata.pkl"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME       = "HuggingFaceH4/zephyr-7b-beta"
RETRIEVE_K     = 3
MAX_NEW_TOKENS = 300
DEVICE         = 0 if torch.cuda.is_available() else "cpu"


# In[ ]:


# Load FAISS index and metadata
faiss_index = faiss.read_index(INDEX_PATH)
doc_meta    = pickle.load(open(META_PATH, "rb"))
print(f"✅ FAISS index loaded with {faiss_index.ntotal} vectors")


# In[ ]:


# Embed query + retrieve top K
def retrieve(query: str, k: int = RETRIEVE_K, embedder=None):
    embedder = embedder or SentenceTransformer(EMBED_MODEL)
    q_vec = embedder.encode(query, normalize_embeddings=True)
    D, I  = faiss_index.search(q_vec.reshape(1, -1).astype("float32"), k)
    return [dict(doc_meta[i], sim=float(score)) for score, i in zip(D[0], I[0])]


# In[ ]:


# Prompt builder
SYSTEM_PROMPT = (
    "You are an educational planning assistant. "
    "Use only the CONTEXT provided. Cite facts as [SOURCE:<section>]."
)

def make_prompt(question: str, docs: list[dict]) -> str:
    context = "\n\n---\n".join(
        f"{doc['text']} [SOURCE:{doc['section']}]"
        for doc in docs
    )
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n<|assistant|>\n"


# In[ ]:


# Load Zephyr-7B safely on GPU (if available)
generator = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ),
    tokenizer=AutoTokenizer.from_pretrained(
        LLM_NAME,
        use_fast=True,
        trust_remote_code=True
    ),
    max_new_tokens=MAX_NEW_TOKENS
)


# In[ ]:


# RAG pipeline with token-safe truncation and attention mask
def rag_chat(question: str, k: int = RETRIEVE_K, max_new_tokens: int = MAX_NEW_TOKENS):
    hits = retrieve(question, k=k)
    prompt = make_prompt(question, hits)

    tokenizer = generator.tokenizer
    model_max_len = 8192
    input_max_len = model_max_len - max_new_tokens

    encodings = tokenizer(prompt, return_tensors="pt", truncation=True,
                          padding=True, max_length=input_max_len)
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


# # Section 3: Prompt Engineering

# In[ ]:


# RAG Goal Generation Pipeline – Gradio-Ready

# --- Prompt Builders ---

def extract_student_info_prompt(profile_text):
    return f"""Extract the following structured information from the student's profile text below.
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

SYSTEM_PROMPT = (
    "You are an educational planning assistant. "
    "Use only the CONTEXT provided. Cite facts as [SOURCE:<section>]."
)

def make_prompt(question: str, docs: list[dict]) -> str:
    context = "\n\n---\n".join(
        f"{doc['text']} [SOURCE:{doc['section']}]" for doc in docs
    )
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n<|assistant|>\n"

# --- Mock LLM + Search (Replace in production) ---

def llm(prompt: str) -> dict:
    print("LLM PROMPT:")
    print(prompt)
    return {
        "employment_goal": "After high school, Clarence will obtain a full-time job at Walmart as a sales associate.",
        "education_goal": "After high school, Clarence will complete on-the-job training provided by Walmart and participate in employer-sponsored customer service workshops.",
        "annual_goal": "In 36 weeks, Clarence will demonstrate effective workplace communication and customer service skills...",
        "alignment": ["OOH standards for Retail Sales Workers", "21st Century Skills"],
        "benchmarks": [
            "Greet customers appropriately",
            "Maintain eye contact",
            "Listen actively",
            "Respond to customer questions"
        ]
    }

def vector_search(query: str) -> list:
    return [
        {"text": "Retail sales roles require strong communication, customer service, and product knowledge.", "section": "OOH-Retail"},
        {"text": "21st Century standards emphasize collaboration, communication, and critical thinking.", "section": "OR-21stCentury"}
    ]

# --- Main Processing Pipeline ---

def process_student_profile(profile_text: str):
    # Step 1: Extract structured fields
    extraction_prompt = extract_student_info_prompt(profile_text)
    structured_info = llm(extraction_prompt)

    # Step 2: Search vector DB using postsecondary employment goal
    goal_query = structured_info.get("postsecondary goal (employment)", "undecided")
    docs = vector_search(goal_query)

    # Step 3: Generate SMART goal prompt
    question = generate_goals_prompt(structured_info)
    full_prompt = make_prompt(question, docs)

    # Step 4: Generate final goals
    return llm(full_prompt)

# --- Example Run (Clarence Case Study) ---

if __name__ == "__main__":
    clarence_case_study = """
    Clarence is a 15-year-old sophomore with a behavior disorder.
    He completed the O*Net Interest Profiler and showed strong interest in the 'Enterprising' category.
    Career interests include retail sales and driver/sales worker.
    Clarence prefers hands-on learning over academic instruction.
    He expressed in his Vision for the Future interview that he would like to work at Walmart.
    """
    results = process_student_profile(clarence_case_study)
    for k, v in results.items():
        print(f"{k.upper()}:\n{v}\n")
