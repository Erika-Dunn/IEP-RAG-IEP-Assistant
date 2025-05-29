import gradio as gr
from FINAL_NLP_Course_CLEAN import process_student_profile

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

def generate_goals(student_name, grade, career, strengths, needs):
    profile_text = f"""
    {student_name} is in grade {grade} and is interested in {career}.
    Strengths: {strengths}
    Needs: {needs}
    """

    try:
        result = process_student_profile(profile_text)
    except Exception as e:
        return f"❌ Error: {e}"

    # Format lists with fallbacks
    benchmarks = result.get("benchmarks") or []
    alignment = result.get("alignment") or []

    benchmarks_str = "- " + "\n- ".join(benchmarks) if benchmarks else "None"
    alignment_str = "- " + "\n- ".join(alignment) if alignment else "None"

    return f"""
🎯 Employment Goal:
{result.get('employment_goal', 'N/A')}

📘 Education Goal:
{result.get('education_goal', 'N/A')}

📝 Annual Goal:
{result.get('annual_goal', 'N/A')}

📌 Benchmarks:
{benchmarks_str}

📎 Alignment:
{alignment_str}
"""

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 IEP Goal Generator")

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

# ✅ Public URL for Colab
demo.launch(share=True)
