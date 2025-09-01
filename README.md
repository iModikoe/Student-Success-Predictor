# Student Success Predictor

Predict whether a student is likely to pass (or their grade band) using features like study habits, attendance, prior grades, and family support.

## Why this matters

_As a tutor, I’ve seen how small changes in study habits and consistency can impact performance. This project explores how data science can help identify at‑risk students early and provide the right support to help them succeed._

## Project Structure

```
student-success-predictor/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   └── (saved models / schema)
├── data/
│   ├── sample_students.csv
│   └── README.txt
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_FeatureEngineering.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Insights.ipynb
├── src/
│   └── utils.py
├── requirements.txt
└── README.md
```

## Dataset Options

- **UCI Student Performance Dataset** (grades, study time, failures, absences, family support, etc.).
- **Kaggle** datasets such as _Student Exam Performance_ or _Student Alcohol Consumption vs. Performance_.

This repo ships with a small synthetic dataset at `data/sample_students.csv` so you can run everything end‑to‑end immediately. Replace it later with a real dataset (UCI or Kaggle).

## Quickstart

```bash
# 1) Create env and install deps
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Explore and train
jupyter lab  # or jupyter notebook
# run notebooks/03_Modeling.ipynb to train & save best model to artifacts/

# 3) Run the app
streamlit run app/streamlit_app.py
```

## Modeling Approach

- EDA → Feature Engineering → Baseline Models (LogReg, Decision Tree) → Improved Models (RandomForest, XGBoost)
- Evaluation: Accuracy, Precision, Recall, F1. Class balance handled via class weights if needed.
- Save the best model as a single `Pipeline` (`artifacts/best_model.joblib`) plus a feature `schema.json` used by the Streamlit app.

## Replacing the Dataset

1. Put your CSV in `data/your_file.csv`.
2. Update the `DATA_PATH` variable in the notebooks (cells at the top) to point to your file.
3. Ensure your target column is `passed` (0/1). If you have `final_grade` (0-20), create `passed = (final_grade >= 10).astype(int)`.

## License

MIT


## Included Real Datasets

- `data/student-mat.csv` → Math subject dataset from UCI Student Performance.
- `data/student-por.csv` → Portuguese subject dataset from UCI Student Performance.
- Both have `G1`, `G2`, `G3` (grades), study time, absences, support, etc.

Tip: To create a `passed` column, use:
```python
df['passed'] = (df['G3'] >= 10).astype(int)
```


---

## Running on the UCI Subjects Separately

This project includes the UCI Student Performance CSVs:

- `data/student-mat.csv` (Math)
- `data/student-por.csv` (Portuguese)

Use the subject-specific notebooks to keep results separate and avoid overwriting artifacts:

- `notebooks/01_EDA_MAT.ipynb` / `01_EDA_POR.ipynb`
- `notebooks/02_FeatureEngineering_MAT.ipynb` / `02_FeatureEngineering_POR.ipynb`
- `notebooks/03_Modeling_MAT.ipynb` → saves to `artifacts/math/`
- `notebooks/03_Modeling_POR.ipynb` → saves to `artifacts/portuguese/`

Both modeling notebooks automatically create `passed` from `G3` when present: `passed = (G3 >= 10)`.
