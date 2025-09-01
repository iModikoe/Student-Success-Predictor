# app/streamlit_app.py
import json
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------- Page ----------
st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Success Predictor")

st.markdown("""
This app predicts whether a student is **likely to pass or fail their final exam** 
based on their early grades, attendance, study habits, and lifestyle factors.  
It was trained on the **UCI Student Performance dataset** using machine learning models 
(Logistic Regression, Random Forest, XGBoost).

👉 How it works:
- You enter details about a student (grades, absences, habits, etc.).
- The model analyzes the inputs.
- It outputs a **PASS/FAIL prediction** along with the probability of passing.

⚡ Purpose: To help identify **at-risk students early** so teachers, tutors, or the 
students themselves can take action before final exams.
""")

# ---------- Friendly labels & help ----------
LABELS = {
    "G1": "First period grade (0–20)",
    "G2": "Second period grade (0–20)",
    "G3": "Final grade (0–20, not used for mid-course predictions)",
    "absences": "Total absences (days)",
    "Dalc": "Workday alcohol use (1=very low … 5=very high)",
    "Walc": "Weekend alcohol use (1=very low … 5=very high)",
    "freetime": "Free time after school (1=very low … 5=very high)",
    "goout": "Going out with friends (1=very low … 5=very high)",
    "health": "Self-reported health (1=very bad … 5=very good)",
}

HELPS = {
    "G1": "Grade before midterm; strong predictor of final success.",
    "G2": "Grade after midterm; stronger predictor than G1 if available.",
    "absences": "Higher absences usually reduce pass chances.",
    "Dalc": "Alcohol use Mon–Fri.",
    "Walc": "Alcohol use on weekends.",
    "freetime": "Balance is helpful; extremes may reflect less study.",
    "goout": "Social time—moderation helps.",
    "health": "Overall health status.",
}

# Optional: bounds for common numeric fields (keeps inputs sane)
RANGES = {
    "G1": (0, 20, 10),
    "G2": (0, 20, 10),
    "G3": (0, 20, 10),
    "absences": (0, 100, 3),
    "Dalc": (1, 5, 2),
    "Walc": (1, 5, 2),
    "freetime": (1, 5, 3),
    "goout": (1, 5, 3),
    "health": (1, 5, 3),
}

# ---------- Subject & artifacts ----------
subject = st.selectbox("Choose subject", ["Math", "Portuguese"])
ART = (Path(__file__).resolve().parents[1] / "artifacts" /
       ("math" if subject == "Math" else "portuguese"))
model_path = ART / "best_model.joblib"
schema_path = ART / "feature_schema.json"

if not model_path.exists() or not schema_path.exists():
    st.error(f"No trained model found for **{subject}**.\n\n"
             f"Please run the `{('03_Modeling_MAT.ipynb' if subject=='Math' else '03_Modeling_POR.ipynb')}` notebook first.")
    st.stop()

pipe = joblib.load(model_path)
with open(schema_path, "r", encoding="utf-8") as f:
    schema = json.load(f)

st.success(f"✅ Loaded **{subject}** model.")

# ---------- Presets (fill the form with sensible examples) ----------
st.subheader("Try example scenarios")
c1, c2 = st.columns(2)
if c1.button("✅ Likely Pass preset"):
    st.session_state.update({"G1": 16, "G2": 17, "absences": 2, "freetime": 3, "health": 4, "Dalc": 1, "Walc": 1})
if c2.button("⚠️ At Risk preset"):
    st.session_state.update({"G1": 6, "G2": 7, "absences": 18, "freetime": 2, "health": 2, "Dalc": 4, "Walc": 4})

# ---------- Form ----------
st.subheader("Enter Student Details")

with st.form("student_form"):
    inputs = {}

    # Group fields into simple sections
    with st.expander("📘 Study & Grades", expanded=True):
        for col in schema:
            name, typ, cats = col["name"], col["type"], col.get("categories")
            # keep grades & study-ish fields in this group
            if name not in {"G1", "G2", "G3", "studytime", "traveltime"}:
                continue
            label = LABELS.get(name, name)
            help_txt = HELPS.get(name)

            if typ == "numeric":
                if name in RANGES:
                    lo, hi, default = RANGES[name]
                    val = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                          value=float(st.session_state.get(name, default)),
                                          step=1.0, help=help_txt, key=name)
                else:
                    val = st.number_input(label, value=float(st.session_state.get(name, 0.0)),
                                          step=1.0, help=help_txt, key=name)
                inputs[name] = val
            else:
                if cats:
                    val = st.selectbox(label, options=cats, index=0, help=help_txt, key=name)
                else:
                    val = st.text_input(label, value=str(st.session_state.get(name, "")),
                                        help=help_txt, key=name)
                inputs[name] = val

    with st.expander("📅 Attendance"):
        for col in schema:
            name, typ = col["name"], col["type"]
            if name != "absences":
                continue
            label = LABELS.get(name, name); help_txt = HELPS.get(name)
            lo, hi, default = RANGES.get(name, (0, 365, 0))
            val = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                  value=float(st.session_state.get(name, default)),
                                  step=1.0, help=help_txt, key=name)
            inputs[name] = val

    with st.expander("🩺 Wellbeing & Lifestyle"):
        for col in schema:
            name, typ = col["name"], col["type"]
            if name not in {"Dalc", "Walc", "freetime", "goout", "health"}:
                continue
            label = LABELS.get(name, name); help_txt = HELPS.get(name)
            lo, hi, default = RANGES.get(name, (1, 5, 3))
            # sliders feel better for 1–5 Likert scales
            val = st.slider(label, min_value=int(lo), max_value=int(hi),
                            value=int(st.session_state.get(name, default)),
                            help=help_txt, key=name)
            inputs[name] = val

    # Render any remaining features that weren't covered above
    covered = set(inputs.keys())
    other_cols = [c for c in schema if c["name"] not in covered]
    if other_cols:
        with st.expander("⚙️ Other details"):
            for col in other_cols:
                name, typ, cats = col["name"], col["type"], col.get("categories")
                label = LABELS.get(name, name)
                help_txt = HELPS.get(name)
                if typ == "numeric":
                    val = st.number_input(label, value=float(st.session_state.get(name, 0.0)),
                                          step=1.0, help=help_txt, key=name)
                else:
                    if cats:
                        val = st.selectbox(label, options=cats, index=0, help=help_txt, key=name)
                    else:
                        val = st.text_input(label, value=str(st.session_state.get(name, "")),
                                            help=help_txt, key=name)
                inputs[name] = val

    submitted = st.form_submit_button("Predict")

# ---------- Predict ----------
if submitted:
    X = pd.DataFrame([inputs])

    # Make sure categories present as strings (common for one-hot encoders)
    for col in X.columns:
        if isinstance(X[col].iloc[0], str):
            X[col] = X[col].astype("string")

    try:
        # Some models (trees) may not expose predict_proba for certain settings,
        # fall back to decision_function if needed.
        if hasattr(pipe, "predict_proba"):
            proba = float(pipe.predict_proba(X)[0][1])
        elif hasattr(pipe, "decision_function"):
            from sklearn.preprocessing import MinMaxScaler
            raw = float(pipe.decision_function(X)[0])
            # rough normalization to [0,1] for display if needed
            proba = float((raw - (-5)) / (5 - (-5)))
            proba = min(max(proba, 0.0), 1.0)
        else:
            proba = float(pipe.predict(X)[0])  # 0/1
        pred = int(proba >= 0.5)

        st.markdown(
            f"### Result: **{'✅ PASS' if pred==1 else '❌ FAIL'}**  \n"
            f"**Probability of pass:** `{proba:.2%}`"
        )

        with st.expander("What does this mean?"):
            st.markdown(
                "- This is a probability, not a guarantee.\n"
                "- The model is typically most influenced by **G1, G2, and absences**.\n"
                "- To improve chances: focus on consistent study, reduce absences, and keep a balanced lifestyle."
            )

    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.exception(e)

# ---------- Glossary ----------
with st.expander("🗂 Glossary of fields"):
    st.markdown("""
- **G1 / G2** — First / Second period grades (0–20)  
- **absences** — Total days absent  
- **Dalc** — Workday alcohol use (1–5)  
- **Walc** — Weekend alcohol use (1–5)  
- **freetime** — Free time after school (1–5)  
- **goout** — Going out with friends (1–5)  
- **health** — Self-reported health (1–5)  
""")

st.caption("Itumeleng RJ Modikoe")
