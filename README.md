# Student Performance Predictor

**Python · Scikit-learn · Pandas · Flask** | Sep 2025 – Dec 2025

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Supervised ML project predicting secondary school students' final grades (G3/20) using the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008 — 395 students, 33 features).

---

## Live Demo

```
python app/api.py
→ Open http://localhost:5000
```

---

## Results

| Model | Exp A R² | Exp A CV R² | Exp B R² | Exp B CV R² |
|---|---|---|---|---|
| Linear Regression | 0.852 | 0.812 | 0.234 | 0.103 |
| Decision Tree | 0.725 | 0.733 | — | — |
| **Random Forest** | **0.874** | **0.813** | **0.341** | **0.185** |

**Experiment A** — With prior grades G1, G2 (clean dataset, n=359)  
**Experiment B** — Demographics only, no exam scores (full dataset, n=395)

---

## What Makes This Unique

### 1. Two experiments, not one number
Most ML projects report one accuracy metric. This project compares two setups:
- **Exp A** (87% R²): School has mid-year exam data → identify at-risk students
- **Exp B** (34% R²): Enrollment-time → predict from demographics alone

The gap between them tells the real story: prior performance is the dominant signal.

### 2. Feature engineering with educational rationale

| Feature | Formula | Why |
|---|---|---|
| `parent_edu` | `Medu + Fedu` | Combined parental education |
| `study_fail` | `studytime / (failures + 1)` | Effectiveness, not just effort |
| `grade_trend` | `G2 - G1` | Academic momentum |
| `avg_grade` | `(G1 + G2) / 2` | Baseline performance |

### 3. Cross-validation on every model
5-fold CV across the full dataset — not just one 80/20 split. Confirms generalisation.

### 4. Dropout students handled correctly
G3=0 students are real dropouts. Excluded from Exp A for clean regression; included in Exp B for realistic enrollment-time modelling. Most projects silently drop these rows.

### 5. Fully dynamic dashboard
Every chart and KPI fetches live from the Flask API at page load. Retrain → refresh → everything updates automatically.

---

## Project Structure

```
spp/
├── data/
│   ├── student-mat.csv          # Full UCI dataset (395 rows)
│   └── student-mat-clean.csv    # G3>0 only (359 rows)
├── src/
│   ├── preprocess.py            # Load, encode, feature engineering
│   └── train_model.py           # Train all 3 models, save artefacts
├── app/
│   └── api.py                   # Flask REST API (9 endpoints)
├── models/                      # Saved artefacts (auto-generated)
├── notebooks/
│   └── EDA.ipynb                # Full exploratory data analysis
├── dashboard.html               # Dynamic portfolio dashboard
├── requirements.txt
├── Procfile                     # Deployment (Railway / Render)
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/<your-username>/student-performance-ml
cd student-performance-ml

pip install -r requirements.txt

python src/train_model.py    # train models → saves to models/
python app/api.py            # start Flask on localhost:5000
# → open http://localhost:5000
```

To run the EDA notebook:
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive dashboard |
| `GET` | `/api` | API info & usage |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | All model metrics |
| `GET` | `/features/<mode>` | Feature list |
| `GET` | `/dataset/stats` | Dataset summary |
| `GET` | `/dataset/corr` | Correlation matrix |
| `GET` | `/dataset/grades` | Grade distribution |
| `GET` | `/dataset/scatter` | Actual vs predicted |
| `POST` | `/predict` | Predict G3 grade |

### Predict Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "with_grades",
    "features": {
      "G1": 12, "G2": 13, "studytime": 2, "failures": 0,
      "absences": 4, "Medu": 3, "Fedu": 2, "higher": 1,
      "parent_edu": 5, "study_fail": 2.0,
      "grade_trend": 1, "avg_grade": 12.5
    }
  }'
```

**Response:**
```json
{
  "predicted_grade": 12.8,
  "letter_grade": "C",
  "description": "Pass",
  "percentage": 64.0,
  "model": "Random Forest (n_estimators=300, max_depth=10)"
}
```

---

## Deployment

### Railway (recommended — free)
1. Push to GitHub
2. New project → Deploy from GitHub repo
3. Set `PORT` env variable → Railway auto-detects `Procfile`

### Render
1. New Web Service → connect GitHub repo
2. Build command: `pip install -r requirements.txt && python src/train_model.py`
3. Start command: `gunicorn --chdir app api:app`

---

## Reference

> P. Cortez and A. Silva (2008).
> *Using Data Mining to Predict Secondary School Student Performance.*
> Proc. 5th Annual Future Business Technology Conf., Porto, pp. 5–12.
> UCI ML Repository: https://archive.ics.uci.edu/dataset/320/student+performance
