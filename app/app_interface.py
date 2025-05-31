# app_interface.py

import gradio as gr
from FINAL_NLP_Course_CLEAN import process_student_profile

def explain_results(result_dict):
    return (
        f"### Measurable Postsecondary Goals\n"
        f"- **Employment:** {result_dict.get('employment_goal', 'N/A')}\n"
        f"- **Education/Training:** {result_dict.get('education_goal', 'N/A')}\n\n"
        f"### Annual Goal\n"
        f"{result_dict.get('annual_goal', 'N/A')}\n\n"
        f"### Standards Alignment\n"
        + "".join(f"- {s}\n" for s in result_dict.get('alignment', [])) +
        f"\n### Benchmarks/Objectives\n"
        + "".join(f"- {b}\n" for b in result_dict.get('benchmarks', []))
    )

def run_pipeline(profile_text):
    result = process_student_profile(profile_text)
    return explain_results(result)

iface = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Textbox(lines=12, label="Student Profile", placeholder="Paste or type the student's background, interests, and assessment results..."),
    outputs=gr.Markdown(label="Generated IEP Goals & Standards Alignment"),
    title="IEP Goal Generator",
    description="Enter a student profile and receive aligned postsecondary and annual IEP goals based on interests, assessments, and standards."
)

if __name__ == "__main__":
    iface.launch(share=True)
