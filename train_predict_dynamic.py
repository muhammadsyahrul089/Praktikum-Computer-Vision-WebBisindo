import pandas as pd
import numpy as np
import pickle
import os
import cv2
import time
import mediapipe as mp
from collections import Counter, deque

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import string

# =========================
# LOAD DATA
# =========================
dataset_dir = "dataset"

if not os.path.exists(dataset_dir):
    print("Folder dataset tidak ditemukan!")
    exit()

# Baca semua CSV dari dataset/huruf/, dataset/angka/, dataset/kata/
all_dfs = []
for kategori in ["huruf", "angka", "kata"]:
    folder = os.path.join(dataset_dir, kategori)
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            if os.path.getsize(path) > 0:
                all_dfs.append(pd.read_csv(path, header=None, dtype={0: str}))

if not all_dfs:
    print("Tidak ada data di folder dataset!")
    exit()

df = pd.concat(all_dfs, ignore_index=True)
df.iloc[:, 0] = df.iloc[:, 0].astype(str)

print("Jumlah data:", len(df))

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=300))
])

print("\nTraining model...")
model.fit(X_train, y_train)

# =========================
# EVALUASI
# =========================
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Akurasi: {acc*100:.2f}%")

# =========================
# SIMPAN MODEL
# =========================
with open("model_dynamic.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model disimpan!")