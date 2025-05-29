# 🧠 RAG-IEP: Retrieval-Augmented Generation for IEP Goal Development

A Retrieval-Augmented Generation (RAG) system that helps educators draft compliant, personalized IEP goals for students with disabilities. The system synthesizes data from occupational outlooks, educational standards, and best-practice IEP templates to generate actionable, standards-aligned goals using large language models.

---

## 🚀 Project Overview

This project was developed as a culminating assignment to demonstrate proficiency in RAG pipelines, educational NLP applications, and synthetic data integration. It pulls from three data sources:

- **Occupational Outlook Handbook** (BLS.gov)
- **Oregon State Educational Standards** (Career Readiness & 21st Century Skills)
- **Sample IEP Goals** from NASET

---

## 📂 Project Structure

```bash
rag-iep/
├── app/
│   ├── FINAL_NLP_Course_CLEAN.py
│   ├── app_interface.py
│   └── data/
├── docs/
│   └── analysis.md
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt
```

---

## 🛠️ How to Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run the main script**

```bash
python app/FINAL_NLP_Course_CLEAN.py
```

3. **Use the interface**

If applicable, run:

```bash
python app/app_interface.py
```

---

## 💡 Features

- 🔍 **Semantic search** with FAISS and `sentence-transformers`
- 🧱 **Hierarchical chunking** of structured educational content
- 🤖 **Prompt engineering** for educational use cases
- 📝 **Goal generation** aligned to occupational and educational standards

---

## 📊 Evaluation

| Metric                  | Result                |
|-------------------------|------------------------|
| Relevance Accuracy      | ~85%                   |
| Goal Quality (manual)   | ~85% pass rate         |
| End-to-End Latency      | <3 sec/query (Colab Pro) |

---

## 📌 Future Work

- Fine-tune generation models on domain-specific IEP data
- Expand to support multiple state standards and job databases
- Build a lightweight educator-facing UI for interactive usage
- Integrate user feedback for continuous improvement

---

## 📄 Documentation

Full methodology and analysis available in [`docs/analysis.md`](docs/analysis.md)

---

## 🤝 Acknowledgments

- HuggingFace Transformers and Sentence Transformers
- Oregon Department of Education (open educational standards)
- Bureau of Labor Statistics for career data
- National Association of Special Education Teachers (NASET)

---

## 📜 License

MIT License (see `LICENSE` file)

---

## 🧠 Author

Developed by **Erika Kelly**  
Data Scientist | Experimentation Specialist | Education Enthusiast
