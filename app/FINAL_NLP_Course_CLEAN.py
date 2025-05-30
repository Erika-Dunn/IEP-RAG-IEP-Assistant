from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

def build_prompt(student_info, retrieved_chunks):
    return f"""You are an expert in creating Individualized Education Program (IEP) goals.
Based on the student profile and aligned standards below, generate:

- One measurable Postsecondary Employment Goal
- One measurable Education/Training Goal
- One Annual Goal aligned with state standards
- Three short-term Objectives supporting the annual goal

### Student Profile:
{student_info}

### Standards and Career Info:
{retrieved_chunks}

### Format:
Employment Goal: ...
Education Goal: ...
Annual Goal: ...
Objectives:
1. ...
2. ...
3. ...
"""

def generate_iep_goals(student_info, retrieved_docs):
    retrieved_text = "\n".join([doc.get("content", "") for doc in retrieved_docs])
    prompt = build_prompt(student_info, retrieved_text)
    response = llm_pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].strip()
