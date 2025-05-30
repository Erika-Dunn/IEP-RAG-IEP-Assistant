# app_interface.py
# ✅ Gradio interface using local Hugging Face pipeline

import gradio as gr
from FINAL_NLP_Course_CLEAN import process_student_profile
import os
import warnings

# Suppress TF and tokenizer warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# --- Wrapper function for Gradio ---
def generate_goals(student_name, grade, career, strengths, needs):
    profile_text = f"""
    {student_name} is in grade {grade} and is interested in {career}.
    Strengths: {strengths}
    Needs: {needs}
    """
    result = process_student_profile(profile_text)

    # ✅ NEW FORMAT: Zephyr-style JSON output
    if all(k in result for k in ["academic_goal", "independent_living_goal", "career_preparation_goal"]):
        return f"""
🎯 Academic Goal:
{result['academic_goal']}

🏠 Independent Living Goal:
{result['independent_living_goal']}

💼 Career Preparation Goal:
{result['career_preparation_goal']}
"""

    # 🟡 FALLBACK: Older OpenAI-style structured fields
    elif any(k in result for k in ["employment_goal", "annual_goal", "raw_output"]):
        if "raw_output" in result:
            return f"📄 Raw Response from LLM:\n\n{result['raw_output']}"

        benchmarks = "- " + "\n- ".join(result.get("benchmarks", [])) if isinstance(result.get("benchmarks"), list) else result.get("benchmarks", "N/A")
        alignment = "- " + "\n- ".join(result.get("alignment", [])) if isinstance(result.get("alignment"), list) else result.get("alignment", "N/A")

        return f"""
🎯 Employment Goal:
{result.get('employment_goal', 'N/A')}

📘 Education Goal:
{result.get('education_goal', 'N/A')}

📝 Annual Goal:
{result.get('annual_goal', 'N/A')}

📌 Benchmarks:
{benchmarks}

📎 Alignment:
{alignment}
"""

    # 🔴 If the LLM output is unstructured or unusable
    return "⚠️ Unexpected output format. Please check the input or try again."

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 IEP Goal Generator (Local)")

    with gr.Row():
        name = gr.Textbox(label="Student Name")
        grade = gr.Dropdown(choices=["8", "9", "10", "11", "12"], label="Grade Level")

    career = gr.Textbox(label="Career Interest")
    skills = gr.Textbox(label="Student Strengths")
    needs = gr.Textbox(label="Student Needs or Accommodations")

    generate_btn = gr.Button("Generate IEP Goals")
    output = gr.Textbox(label="Generated IEP Goals", lines=15)

    generate_btn.click(
        fn=generate_goals,
        inputs=[name, grade, career, skills, needs],
        outputs=output
    )

# Launch in Colab with public URL
if __name__ == "__main__":
    demo.launch(share=True)
