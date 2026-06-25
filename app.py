from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os
import string

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "model_dynamic.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"\nModel '{MODEL_PATH}' tidak ditemukan!")
    print("Jalankan train_predict_dynamic.py terlebih dahulu!")
    model = None
else:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model berhasil dimuat!")
    except Exception as e:
        print(f"Gagal load model: {e}")
        model = None

# =========================
# AMBIL LABEL DARI DATASET FOLDER
# =========================
dataset_dir = "dataset"
all_labels  = set()

for kategori in ["huruf", "angka", "kata"]:
    folder = os.path.join(dataset_dir, kategori)
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            label = os.path.splitext(file)[0]
            all_labels.add(label)

huruf_labels = set(string.ascii_uppercase) & all_labels
angka_labels = {l for l in all_labels if l.isdigit()}
kata_labels  = all_labels - huruf_labels - angka_labels


def get_active_labels(mode):
    if mode == "HURUF":
        return huruf_labels
    elif mode == "ANGKA":
        return angka_labels
    elif mode == "KATA":
        return kata_labels if kata_labels else all_labels
    else:
        return all_labels


def count_rows_in_csv(filepath):
    """Hitung jumlah baris data (bukan header) dalam file CSV."""
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
        # Kurangi 1 untuk header jika ada
        count = len([l for l in lines if l.strip()])
        # Cek apakah baris pertama adalah header (non-numerik)
        if count > 0:
            first = lines[0].strip().split(",")[0]
            try:
                float(first)
            except ValueError:
                count -= 1  # Ada header, kurangi 1
        return max(count, 0)
    except Exception:
        return 0


# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok" if model else "error"})


@app.route("/labels")
def get_labels():
    return jsonify({
        "huruf": sorted(list(huruf_labels)),
        "angka": sorted(list(angka_labels)),
        "kata": sorted(list(kata_labels)),
        "all": sorted(list(all_labels))
    })


@app.route("/stats")
def get_stats():
    """
    Kembalikan jumlah data (baris CSV) per label per kategori.
    Format response:
    {
      "huruf": {"A": 30, "B": 25, ...},
      "angka": {"0": 30, "1": 28, ...},
      "kata":  {"rumah": 40, "makan": 35, ...},
      "total": 1200
    }
    """
    stats = {"huruf": {}, "angka": {}, "kata": {}, "total": 0}
    grand_total = 0

    for kategori in ["huruf", "angka", "kata"]:
        folder = os.path.join(dataset_dir, kategori)
        if not os.path.exists(folder):
            continue
        for file in sorted(os.listdir(folder)):
            if file.endswith(".csv"):
                label = os.path.splitext(file)[0]
                filepath = os.path.join(folder, file)
                count = count_rows_in_csv(filepath)
                stats[kategori][label] = count
                grand_total += count

    stats["total"] = grand_total
    return jsonify(stats)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model belum dimuat"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body kosong"}), 400

    features = data.get("features")
    mode = data.get("mode", "HURUF").upper()

    if not features:
        return jsonify({"error": "Field 'features' wajib ada"}), 400

    try:
        X = np.array(features, dtype=np.float64).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Format features salah: {str(e)}"}), 400

    expected_len = 20 * 126

    # Auto padding/truncate
    if X.shape[1] < expected_len:
        padding = np.zeros((1, expected_len - X.shape[1]))
        X = np.hstack([X, padding])
    elif X.shape[1] > expected_len:
        X = X[:, :expected_len]

    # Prediksi Langsung
    pred = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    classes = model.classes_.tolist()
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [{"label": str(classes[i]), "confidence": float(proba[i])} for i in top3_idx]

    # Filter sesuai mode
    active = get_active_labels(mode)
    if pred not in active:
        for i in np.argsort(proba)[::-1]:
            if str(classes[i]) in active:
                pred = str(classes[i])
                confidence = float(proba[i])
                break
        else:
            pred = "?"

    return jsonify({
        "prediction": pred,
        "confidence": round(confidence * 100, 1),
        "top3": top3,
        "mode": mode
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Bisindo Web API Started")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)