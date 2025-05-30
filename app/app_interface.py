# app_interface.py

import gradio as gr
from FINAL_NLP_Course_CLEAN import vector_search, generate_iep_goals

# ─────────────────────────────────────────────────────────────────────────────
# Wrapper function Gradio will call
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(student_input):
    if not student_input.strip():
        return "Please enter a student profile."

    try:
        docs = vector_search(student_input, k=3)
        result = generate_iep_goals(student_input, docs)
        return result
    except Exception as e:
        return {"error": "Pipeline failed", "details": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI setup
# ─────────────────────────────────────────────────────────────────────────────

iface = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Textbox(label="Enter Student Profile", lines=10, placeholder="e.g., Clarence is a 15-year-old..."),
    outputs=gr.JSON(label="Generated IEP Goals"),
    title="IEP Goal Generator",
    description="This tool generates measurable IEP goals based on student info and aligned standards using a RAG pipeline.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()

