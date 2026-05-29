# ============================================================
#   KLASIFIKASI KEMATANGAN PISANG — CNN + HSV + KAMERA
#   FINAL VERSION (AKURAT & BALANCED)
# ============================================================

import os
import cv2
import numpy as np
import base64
import threading

from flask import Flask, render_template, Response, jsonify, request

import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

# ============================================================
# KONFIGURASI
# ============================================================

IMG_SIZE = 128
MODEL_PATH = "banana_model.h5"

KELAS = [
    "Mentah",
    "Matang",
    "Terlalu Matang"
]

WARNA_KELAS = {
    "Mentah": "#4CAF50",
    "Matang": "#fff242",
    "Terlalu Matang": "#8B4513",
    "Tidak Yakin": "#888888"
}

app = Flask(__name__)

model = None
kamera = None
lock = threading.Lock()

# ============================================================
# MODEL CNN
# ============================================================

def buat_model():

    model = Sequential([

        Conv2D(
            32,
            (3,3),
            activation='relu',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(3, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ============================================================
# DATA DUMMY BALANCED
# ============================================================

def buat_data_dummy(n=250):

    X = []
    y = []

    warna = [

        # MENTAH - HIJAU
        # HEX #4CAF50
        (55, 5, 170, 20, 170, 20),

        # MATANG - KUNING
        # HEX #fff242
        (26, 4, 210, 18, 235, 15),

        # TERLALU MATANG - COKLAT
        # HEX #8B4513
        (12, 3, 150, 20, 115, 18),
    ]

    for label, (h_mu, h_std, s_mu, s_std, v_mu, v_std) in enumerate(warna):

        for _ in range(n):

            H = np.clip(
                np.random.normal(h_mu, h_std, (IMG_SIZE, IMG_SIZE)),
                0,
                179
            )

            S = np.clip(
                np.random.normal(s_mu, s_std, (IMG_SIZE, IMG_SIZE)),
                0,
                255
            )

            V = np.clip(
                np.random.normal(v_mu, v_std, (IMG_SIZE, IMG_SIZE)),
                0,
                255
            )

            hsv = np.stack([H, S, V], axis=-1)

            hsv = hsv.astype("float32") / 255.0

            X.append(hsv)
            y.append(label)

    return np.array(X), np.array(y)

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path="dataset"):

    X = []
    y = []

    folder_kelas = [
        "mentah",
        "matang",
        "terlalu_matang"
    ]

    ekstensi_ok = (
        '.jpg',
        '.jpeg',
        '.png',
        '.webp'
    )

    for label, folder in enumerate(folder_kelas):

        folder_path = os.path.join(path, folder)

        if not os.path.exists(folder_path):
            continue

        for fname in os.listdir(folder_path):

            if not fname.lower().endswith(ekstensi_ok):
                continue

            img_path = os.path.join(folder_path, fname)

            img = cv2.imread(img_path)

            if img is None:
                print(f"[WARN] Tidak bisa membaca {fname}")
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # blur supaya noise hilang
            img = cv2.GaussianBlur(img, (5,5), 0)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # masking area pisang
            lower = np.array([5, 40, 40])
            upper = np.array([80, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)

            kernel = np.ones((5,5), np.uint8)

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

            hsv = hsv.astype("float32") / 255.0

            X.append(hsv)
            y.append(label)

    return np.array(X), np.array(y)

# ============================================================
# TRAIN / LOAD MODEL
# ============================================================

def init_model():

    global model

    if os.path.exists(MODEL_PATH):

        print("[INFO] Memuat model lama...")
        model = load_model(MODEL_PATH)
        print("[INFO] Model berhasil dimuat")
        return

    print("[INFO] Melatih model baru...")

    if os.path.exists("dataset"):

        X_real, y_real = load_dataset("dataset")

        print(f"[INFO] Dataset asli: {len(X_real)}")

        X_dummy, y_dummy = buat_data_dummy(n=250)

        print(f"[INFO] Dummy dataset: {len(X_dummy)}")

        X = np.concatenate([X_real, X_dummy], axis=0)
        y = np.concatenate([y_real, y_dummy], axis=0)

    else:

        X, y = buat_data_dummy(n=250)

    print(f"[INFO] Total dataset: {len(X)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_train_cat = to_categorical(y_train, 3)
    y_val_cat = to_categorical(y_val, 3)

    augmentasi = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15]
    )

    augmentasi.fit(X_train)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    model = buat_model()

    history = model.fit(
        augmentasi.flow(X_train, y_train_cat, batch_size=16),
        validation_data=(X_val, y_val_cat),
        epochs=60,
        callbacks=[early_stop, reduce_lr],
        class_weight={
            0: 1.2,
            1: 1.3,
            2: 1.0
        },
        verbose=1
    )

    loss, acc = model.evaluate(X_val, y_val_cat, verbose=0)

    print(f"[INFO] Akurasi validasi: {acc*100:.2f}%")

    model.save(MODEL_PATH)

    print(f"[INFO] Model disimpan: {MODEL_PATH}")


# ============================================================
# PREDIKSI FRAME (FINAL COLOR ACCURATE)
# ============================================================

def prediksi_frame(frame_bgr):

    # Resize
    img = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))

    # Blur
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Convert HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ========================================================
    # MASK AREA PISANG
    # ========================================================

    # ambil warna hijau-kuning-coklat saja
    lower = np.array([8, 40, 40])
    upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    hasil = cv2.bitwise_and(hsv, hsv, mask=mask)

    # ========================================================
    # CNN PREDIKSI
    # ========================================================

    inp = hasil.astype("float32") / 255.0
    inp = np.expand_dims(inp, axis=0)

    prob = model.predict(inp, verbose=0)[0]

    # ========================================================
    # HSV ANALYSIS
    # ========================================================

    if np.any(mask > 0):

        h_mean = np.mean(hsv[:,:,0][mask > 0])
        s_mean = np.mean(hsv[:,:,1][mask > 0])
        v_mean = np.mean(hsv[:,:,2][mask > 0])

    else:

        h_mean = 0
        s_mean = 0
        v_mean = 0

    # ========================================================
    # RESET BIAS
    # ========================================================

    prob = prob * 0.6

    # ========================================================
    # MENTAH = HIJAU
    # ========================================================

    # hue hijau
    if 40 <= h_mean <= 85:

        prob[0] += 1.20
        prob[1] -= 0.40
        prob[2] -= 0.50

    # ========================================================
    # MATANG = KUNING
    # ========================================================

    # kuning terang
    if 20 <= h_mean <= 38:

        # kuning harus cerah
        if s_mean >= 90 and v_mean >= 140:

            prob[1] += 1.35

            prob[0] -= 0.20
            prob[2] -= 0.60

    # ========================================================
    # TERLALU MATANG = COKLAT
    # ========================================================

    # coklat lebih gelap
    if 5 <= h_mean <= 18:

        # HARUS GELAP
        if v_mean < 120:

            prob[2] += 1.25

            prob[1] -= 0.50

    # ========================================================
    # CLIP NEGATIVE
    # ========================================================

    prob = np.clip(prob, 0, None)

    # ========================================================
    # NORMALISASI
    # ========================================================

    prob = prob / np.sum(prob)

    idx = int(np.argmax(prob))

    # ========================================================
    # CONFIDENCE CHECK
    # ========================================================

    if prob[idx] < 0.55:

        return {
            "kelas": "Tidak Yakin",
            "confidence": float(prob[idx] * 100),
            "probabilitas": {
                KELAS[i]: float(prob[i] * 100)
                for i in range(3)
            },
            "warna": "#888888",
            "h_mean": float(h_mean),
            "s_mean": float(s_mean),
            "v_mean": float(v_mean)
        }

    return {
        "kelas": KELAS[idx],
        "confidence": float(prob[idx] * 100),
        "probabilitas": {
            KELAS[i]: float(prob[i] * 100)
            for i in range(3)
        },
        "warna": WARNA_KELAS[KELAS[idx]],
        "h_mean": float(h_mean),
        "s_mean": float(s_mean),
        "v_mean": float(v_mean)
    }

# ============================================================
# KAMERA
# ============================================================

def get_kamera():

    global kamera

    if kamera is None or not kamera.isOpened():
        kamera = cv2.VideoCapture(0)

    return kamera

# ============================================================
# STREAM VIDEO
# ============================================================

def generate_frames():

    cam = get_kamera()

    while True:

        with lock:
            success, frame = cam.read()

        if not success:
            break

        hasil = prediksi_frame(frame)

        label = f"{hasil['kelas']} ({hasil['confidence']:.1f}%)"

        warna_map = {
            "Mentah": (80,175,76),
            "Matang": (66,242,255),
            "Terlalu Matang": (19,69,139),
            "Tidak Yakin": (180,180,180)
        }

        warna = warna_map.get(hasil['kelas'], (255,255,255))

        h, w = frame.shape[:2]

        cv2.rectangle(frame, (10,10), (w-10,80), (0,0,0), -1)
        cv2.rectangle(frame, (10,10), (w-10,80), warna, 2)

        cv2.putText(
            frame,
            label,
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            warna,
            2
        )

        _, buffer = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/prediksi_kamera')
def prediksi_kamera():

    cam = get_kamera()

    with lock:
        success, frame = cam.read()

    if not success:
        return jsonify({"error": "kamera tidak tersedia"}), 500

    hasil = prediksi_frame(frame)

    return jsonify(hasil)

@app.route('/prediksi_upload', methods=['POST'])
def prediksi_upload():

    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400

    file = request.files['file']

    data = np.frombuffer(file.read(), np.uint8)

    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "gambar tidak valid"}), 400

    hasil = prediksi_frame(frame)

    _, buf = cv2.imencode('.jpg', frame)

    b64 = base64.b64encode(buf).decode('utf-8')

    hasil['preview'] = f'data:image/jpeg;base64,{b64}'

    return jsonify(hasil)

@app.route('/stop_kamera')
def stop_kamera():

    global kamera

    if kamera and kamera.isOpened():
        kamera.release()
        kamera = None

    return jsonify({"status": "kamera dihentikan"})

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    print('=' * 50)
    print(' KLASIFIKASI KEMATANGAN PISANG')
    print(' CNN + HSV + BALANCED DETECTION')
    print('=' * 50)

    # HAPUS MODEL LAMA JIKA MAU TRAIN ULANG
    # os.remove(MODEL_PATH)

    init_model()

    print('\n[INFO] Server berjalan di:')
    print('http://localhost:5000')

    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000
    )