import cv2
import time
import mediapipe as mp
import csv
import os
import string
import pandas as pd
import numpy as np

# =========================
# INPUT LABEL
# =========================
selected_label = input("Masukkan label (Huruf : 'a-z' / Angka : '0-9' / kata : 'halo'): ").strip()

valid_single = list(string.ascii_uppercase) + [str(i) for i in range(10)]

if selected_label.isdigit():
    pass
elif len(selected_label) == 1 and selected_label.isalpha():
    selected_label = selected_label.upper()
    if selected_label not in valid_single:
        print("❌ Label tidak valid!")
        exit()
else:
    selected_label = selected_label.lower()
    if not selected_label.replace(" ", "").isalpha():
        print("❌ Label kata hanya boleh huruf!")
        exit()

# =========================
# PILIHAN FITUR
# =========================

# Tentukan kategori dan folder berdasarkan label
if selected_label.isdigit():
    kategori = "angka"
elif len(selected_label) == 1 and selected_label.isalpha():
    kategori = "huruf"
else:
    kategori = "kata"

# Buat folder dataset/huruf, dataset/angka, dataset/kata jika belum ada
folder_path  = os.path.join("dataset", kategori)
os.makedirs(folder_path, exist_ok=True)

# Path file CSV: dataset/huruf/A.csv, dataset/angka/5.csv, dataset/kata/halo.csv
dataset_path = os.path.join(folder_path, f"{selected_label}.csv")

existing_count = 0
if os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 0:
    df_check = pd.read_csv(dataset_path, header=None)
    existing_count = len(df_check[df_check[0] == selected_label])

print(f"\nLabel '{selected_label}' saat ini punya {existing_count} data.")
print("\nPilih fitur:")
print("  1. Membuat Data Baru")
print("  2. Memperbarui Data")

pilihan = input("\nMasukkan pilihan (1/2): ").strip()

if pilihan not in ["1", "2"]:
    print("❌ Pilihan tidak valid!")
    exit()

if pilihan == "2":
    if os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 0:
        df = pd.read_csv(dataset_path, header=None)
        df = df[df[0] != selected_label]
        df.to_csv(dataset_path, index=False, header=False)
        print(f"[OK] Data lama untuk label '{selected_label}' telah dihapus")
    else:
        open(dataset_path, 'w').close()
        print(f"[OK] File '{selected_label}.csv' dikosongkan")
    mode_label = "PERBARUI"
else:
    print(f"[OK] Data lama untuk label '{selected_label}' dipertahankan")
    mode_label = "TAMBAH"

file = open(dataset_path, mode='a', newline='')
writer = csv.writer(file)

# =========================
# SETTING
# =========================
TARGET_PER_LABEL  = 30
FRAME_PER_GESTURE = 20
count = 0

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=2)
mp_draw  = mp.solutions.drawing_utils

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

print(f"\nMode: {mode_label}")
print(f"Label: '{selected_label}'")
print(f"Target: {TARGET_PER_LABEL} gesture {FRAME_PER_GESTURE} frame")
print(f"\nTekan SPACE untuk memulai | Q untuk mengakhiri")

# State
STATE_WAIT      = "WAIT"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_RECORDING = "RECORDING"

state           = STATE_WAIT
frame_buffer    = []
countdown_start = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    frame  = cv2.resize(frame, (800, 600))
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_data = None

    if result.multi_hand_landmarks:
        data       = []
        hands_data = [None, None]

        if result.multi_handedness:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                hand_label = result.multi_handedness[idx].classification[0].label

                temp = []
                for lm in hand_landmarks.landmark:
                    temp.extend([lm.x, lm.y, lm.z])

                if hand_label == "Left":
                    hands_data[0] = temp
                else:
                    hands_data[1] = temp

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for h in hands_data:
            if h is None:
                data.extend([0] * 63)
            else:
                data.extend(h)

        base_x, base_y, base_z = data[0], data[1], data[2]
        normalized = []
        for i in range(0, len(data), 3):
            normalized.append(data[i]   - base_x)
            normalized.append(data[i+1] - base_y)
            normalized.append(data[i+2] - base_z)

        current_data = normalized

    # =========================
    # STATE MACHINE
    # =========================
    if state == STATE_COUNTDOWN:
        elapsed   = time.time() - countdown_start
        remaining = 3 - int(elapsed)

        if remaining > 0:
            cv2.putText(frame, str(remaining), (340, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 20)
            cv2.putText(frame, f"Siapkan gerakan ke -{count+1}!",
                        (220, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            state        = STATE_RECORDING
            frame_buffer = []
            print(f"Rekam gerakan ke -{count+1}...")

    elif state == STATE_RECORDING:
        if current_data is not None:
            frame_buffer.append(current_data)
        else:
            frame_buffer.append([0] * 126)

        if len(frame_buffer) >= FRAME_PER_GESTURE:
            flat = []
            for f in frame_buffer:
                flat.extend(f)

            writer.writerow([selected_label] + flat)
            count += 1
            print(f"gerakan {count}/{TARGET_PER_LABEL} tersimpan")

            if count >= TARGET_PER_LABEL:
                print(f"\nSelesai! {TARGET_PER_LABEL} gerakan '{selected_label}' terkumpul")
                cv2.putText(frame, "SELESAI!", (250, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
                cv2.imshow("Collect Dataset Dynamic", frame)
                cv2.waitKey(2000)
                break

            state           = STATE_COUNTDOWN
            countdown_start = time.time()
            frame_buffer    = []

    # =========================
    # DISPLAY
    # =========================
    cv2.rectangle(frame, (0, 0), (800, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Label: {selected_label}  |  {count}/{TARGET_PER_LABEL}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    mode_color = (0, 165, 255) if mode_label == "TAMBAH" else (0, 100, 255)
    cv2.putText(frame, f"Mode: {mode_label}", (600, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    if state == STATE_WAIT:
        cv2.putText(frame, "Tekan SPACE untuk mulai",
                    (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)

    elif state == STATE_RECORDING:
        progress = int((len(frame_buffer) / FRAME_PER_GESTURE) * 760)
        cv2.rectangle(frame, (20, 550), (780, 580), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 550), (20 + progress, 580), (0, 0, 255), -1)
        cv2.putText(frame, f"MEREKAM... {len(frame_buffer)}/{FRAME_PER_GESTURE} frame",
                    (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Collect Dataset Dynamic", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' ') and state == STATE_WAIT:
        state           = STATE_COUNTDOWN
        countdown_start = time.time()
        print("Countdown dimulai...")

cap.release()
file.close()
cv2.destroyAllWindows()
print("Dataset dinamis diperbarui!")