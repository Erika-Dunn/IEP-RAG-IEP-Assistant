from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_name = "declare-lab/flan-alpaca-base"  # lightweight, instruction-tuned

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # Force CPU
    pad_token_id=tokenizer.eos_token_id
)

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

def generate_iep_goals(student_info, retrieved_docs):
    retrieved_text = "\n".join([doc.get("content", "") for doc in retrieved_docs])
    prompt = build_prompt(student_info, retrieved_text)
    response = llm_pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
    return response[0]['generated_text'].split(prompt)[-1].strip()
