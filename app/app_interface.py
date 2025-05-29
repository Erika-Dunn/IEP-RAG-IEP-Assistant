# app_interface.py
# ✅ Optional CLI runner for debugging or command-line usage

from FINAL_NLP_Course_CLEAN import process_student_profile

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

for name, profile in students.items():
    print(f"\n🎓 Generating IEP Goals for: {name}")
    results = process_student_profile(profile)
    for k, v in results.items():
        print(f"{k.upper()}:\n{v}\n")
    print("="*80)

