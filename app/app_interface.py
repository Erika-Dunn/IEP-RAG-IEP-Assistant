import gradio as gr
from FINAL_NLP_Course_CLEAN import process_student_profile

def generate_goals(student_name, grade, career, strengths, needs):
    profile_text = f"""
    {student_name} is in grade {grade} and is interested in {career}.
    Strengths: {strengths}
    Needs: {needs}
    """
    result = process_student_profile(profile_text)

    return f"""
🎯 Employment Goal:
{result['employment_goal']}

📘 Education Goal:
{result['education_goal']}

📝 Annual Goal:
{result['annual_goal']}

📌 Benchmarks:
- {'\\n- '.join(result['benchmarks'])}

📎 Alignment:
- {'\\n- '.join(result['alignment'])}
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

    generate_btn.click(fn=generate_goals, 
                       inputs=[name, grade, career, skills, needs], 
                       outputs=output)

demo.launch()
