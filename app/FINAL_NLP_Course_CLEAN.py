import requests

API_URL = "https://api-inference.huggingface.co/models/declare-lab/flan-alpaca-base"
HF_TOKEN = "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"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def build_prompt(student_info, retrieved_chunks):
    return f"""You are an expert in creating Individualized Education Program (IEP) goals. 
Based on the student profile and aligned standards below, generate:

- One measurable **Postsecondary Employment Goal**
- One **Education/Training Goal**
- One **Annual Goal** aligned with state standards
- Three short-term **Objectives** supporting the annual goal

Ensure the goals are specific, measurable, and relevant to the student.

### Student Profile:
{student_info}

### Aligned Standards and Career Info:
{retrieved_chunks}

### Output Format:
Employment Goal: ...
Education Goal: ...
Annual Goal: ...
Objectives:
1. ...
2. ...
3. ...
"""

def query_hf_model(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"[ERROR {response.status_code}]: {response.text}"
    return response.json()[0]["generated_text"]

def generate_iep_goals(student_info, retrieved_docs):
    retrieved_text = "\n".join([doc.get("content", "") for doc in retrieved_docs])
    prompt = build_prompt(student_info, retrieved_text)
    return query_hf_model(prompt).strip()


