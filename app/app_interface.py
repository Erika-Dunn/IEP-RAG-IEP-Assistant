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
    result = process_student_profile(profile_text)

    # Safely format lists for display
    benchmarks = "- " + "\n- ".join(result.get("benchmarks", []))
    alignment = "- " + "\n- ".join(result.get("alignment", []))

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
