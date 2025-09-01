Place real datasets here if you don't want to use sample_students.csv.

Examples:
- UCI Student Performance Dataset (Math or Portuguese)
- Kaggle: Student Exam Performance, Student Alcohol Consumption vs Performance

Tip: Ensure there's a binary target `passed` (0/1). If your dataset has a final numeric grade (0-20), create:
    passed = (final_grade >= 10).astype(int)
