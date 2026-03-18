"""
api.py  —  Student Performance Predictor · Flask REST API
=========================================================
Run:      python app/api.py
Base URL: http://localhost:5000

  GET  /                    → serves dashboard.html (the visual dashboard)
  GET  /api                 → API info JSON
  GET  /health              → model status
  GET  /metrics             → all model metrics (both experiments)
  GET  /features/<mode>     → feature list
  GET  /dataset/stats       → dataset summary
  GET  /dataset/corr        → correlation matrix
  GET  /dataset/grades      → grade distribution
  GET  /dataset/scatter     → actual vs predicted scatter data
  POST /predict             → predict G3 final grade

Prerequisite: python src/train_model.py
"""

from flask import Flask, request, jsonify, send_from_directory
import pickle, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

# Serve /static/* files (favicon, etc.)
@app.route("/static/<path:filename>")
def static_files(filename):
    import os
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)
app.config["JSON_SORT_KEYS"] = False


# ── Load all artefacts once at startup ───────────────────────
def _pkl(fname):
    p = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}  →  run: python src/train_model.py")
    with open(p, "rb") as f:
        return pickle.load(f)

def _json(fname):
    with open(os.path.join(MODELS_DIR, fname)) as f:
        return json.load(f)

try:
    MODEL_W  = _pkl("model_with.pkl")
    MODEL_WO = _pkl("model_without.pkl")
    FEATS_W  = _pkl("feats_with.pkl")
    FEATS_WO = _pkl("feats_without.pkl")
    RESULTS  = _json("all_results.json")
    STATS    = _json("dataset_stats.json")
    CORR     = _json("corr_matrix.json")
    GDIST    = _json("grade_dist.json")
    SCATTER  = _json("scatter_data.json")
    READY    = True
    print("✅  Models loaded.")
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    READY = False


def grade_letter(g):
    if g >= 16: return "A"
    if g >= 14: return "B"
    if g >= 10: return "C"
    return "D"

# ── CORS helper (so dashboard JS can call /api endpoints) ────
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


# ── Dashboard ─────────────────────────────────────────────────
@app.route("/")
def dashboard():
    """Serve the interactive HTML dashboard."""
    return send_from_directory(BASE_DIR, "dashboard.html")


# ── API info ──────────────────────────────────────────────────
@app.route("/api")
def api_info():
    return jsonify({
        "service":  "Student Performance Predictor API",
        "version":  "2.0",
        "dataset":  "UCI Student Performance — Cortez & Silva (2008)",
        "endpoints": {
            "GET  /":                   "Interactive dashboard",
            "GET  /health":             "Health check",
            "GET  /metrics":            "Model metrics (both experiments)",
            "GET  /features/<mode>":    "Feature list",
            "GET  /dataset/stats":      "Dataset summary",
            "GET  /dataset/corr":       "Correlation matrix",
            "GET  /dataset/grades":     "Grade distribution",
            "GET  /dataset/scatter":    "Actual vs predicted (test set)",
            "POST /predict":            "Predict G3 final grade",
        }
    })


# ── Health ────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok" if READY else "degraded", "models_loaded": READY})


# ── Model metrics ─────────────────────────────────────────────
@app.route("/metrics")
def metrics():
    if not READY:
        return jsonify({"error": "Models not loaded"}), 503
    return jsonify(RESULTS)


# ── Feature lists ─────────────────────────────────────────────
@app.route("/features/<mode>")
def features(mode):
    if mode == "with_grades":
        return jsonify({"mode": mode, "features": FEATS_W,  "count": len(FEATS_W)})
    if mode == "without_grades":
        return jsonify({"mode": mode, "features": FEATS_WO, "count": len(FEATS_WO)})
    return jsonify({"error": "mode must be with_grades or without_grades"}), 400


# ── Dataset endpoints ─────────────────────────────────────────
@app.route("/dataset/stats")
def dataset_stats():
    return jsonify(STATS)

@app.route("/dataset/corr")
def dataset_corr():
    return jsonify(CORR)

@app.route("/dataset/grades")
def dataset_grades():
    return jsonify(GDIST)

@app.route("/dataset/scatter")
def dataset_scatter():
    return jsonify(SCATTER)


# ── Predict ───────────────────────────────────────────────────
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    if not READY:
        return jsonify({"error": "Models not loaded. Run: python src/train_model.py"}), 503

    body  = request.get_json(force=True, silent=True) or {}
    mode  = body.get("mode", "with_grades")

    if mode not in ("with_grades", "without_grades"):
        return jsonify({"error": "mode must be with_grades or without_grades"}), 400

    model = MODEL_W  if mode == "with_grades" else MODEL_WO
    feats = FEATS_W  if mode == "with_grades" else FEATS_WO
    raw   = body.get("features", body)

    try:
        fv = [float(raw.get(f, 0)) for f in feats]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Feature error: {e}"}), 400

    pred  = float(model.predict([fv])[0])
    grade = round(max(0.0, min(20.0, pred)), 2)
    letter = grade_letter(grade)

    return jsonify({
        "mode":            mode,
        "predicted_grade": grade,
        "out_of":          20,
        "percentage":      round(grade / 20 * 100, 1),
        "letter_grade":    letter,
        "description":     {"A":"Excellent","B":"Good","C":"Pass","D":"At Risk"}[letter],
        "model":           "Random Forest (n_estimators=300, max_depth=12)",
        "features_used":   len(feats),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀  Dashboard → http://localhost:{port}")
    print(f"    API info  → http://localhost:{port}/api\n")
    app.run(debug=True, port=port)