# app_interface.py
# ✅ Unified CLI and Gradio interface for testing and UI

import gradio as gr
from FINAL_NLP_Course_CLEAN import process_student_profile

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

# --- Gradio Interface Function ---
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

# --- CLI Sample Student Dictionary ---
students = {
    "Clarence": """
    Clarence is a 15-year-old sophomore with a behavior disorder.
    He completed the O*Net Interest Profiler and showed strong interest in the 'Enterprising' category.
    Career interests include retail sales and driver/sales worker.
    Clarence prefers hands-on learning over academic instruction.
    He expressed in his Vision for the Future interview that he would like to work at Walmart.
    """,
    "Marisol": """
    Marisol is a 17-year-old senior who enjoys caring for animals and is detail-oriented.
    She completed the Interest Profiler and scored high in Social and Realistic domains.
    She wants to be a veterinary assistant and has volunteered at the local animal shelter.
    She needs support in written communication and managing time across multiple assignments.
    """,
    "DeShawn": """
    DeShawn is a 16-year-old junior with ADHD. He is creative and excels in hands-on technical tasks.
    He has expressed interest in automotive repair and has participated in a school-sponsored job shadow at a local mechanic’s shop.
    DeShawn struggles with organization and task completion.
    """,
    "Linh": """
    Linh is a 14-year-old freshman who recently moved to the U.S. and is an English language learner.
    She shows strength in mathematics and visual problem solving. Her interests include graphic design and architecture.
    She needs support with English reading comprehension and academic vocabulary.
    """
}

# --- Run CLI Tests and Launch UI if script is called directly ---
if __name__ == "__main__":
    # CLI test output
    for name, profile in students.items():
        print(f"\n🎓 Generating IEP Goals for: {name}")
        results = process_student_profile(profile)
        for k, v in results.items():
            print(f"{k.upper()}:\n{v}\n")
        print("="*80)

    # Gradio interface
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

    demo.launch(share=True)


