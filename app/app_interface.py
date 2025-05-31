# app_interface.py
import os, sys, warnings, contextlib

# Suppress Python + TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Optional: suppress low-level stderr (e.g., cuDNN/cublas XLA)
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


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

def run_pipeline(name, grade, profile_text):
    full_profile = f"Student name: {name}\nGrade level: {grade}\n{profile_text}"
    result = process_student_profile(full_profile)
    return explain_results(result)

iface = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Textbox(label="Student Name", placeholder="e.g. Clarence"),
        gr.Textbox(label="Grade Level", placeholder="e.g. 10th grade"),
        gr.Textbox(lines=10, label="Student Profile", placeholder="Paste or type the student's background, interests, and assessment results...")
    ],
    outputs=gr.Markdown(label="Generated IEP Goals & Standards Alignment"),
    title="IEP Goal Generator",
    description="Enter a student profile and receive aligned postsecondary and annual IEP goals based on interests, assessments, and standards."
)

if __name__ == "__main__":
   iface.launch(share=True)

