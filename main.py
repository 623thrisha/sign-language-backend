from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque
from spellchecker import SpellChecker
from wordfreq import zipf_frequency

app = Flask(__name__)
CORS(app)

# ---------------- GLOBAL ----------------
recognized_text = ""
camera_running = False
cap = None

last_prediction = None
stable_count = 0
last_added_time = 0
last_backspace_time = 0

# ---------------- CONTROL ----------------
STABLE_THRESHOLD = 20
ADD_DELAY = 1.8
BACKSPACE_REPEAT_DELAY = 0.6

# ---------------- SMOOTHING ----------------
prediction_buffer = deque(maxlen=10)

# ---------------- NLP ----------------
spell = SpellChecker()

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ---------------- API ----------------
@app.route("/text")
def get_text():
    return jsonify({
        "text": recognized_text,
        "prediction": predict_next_word(recognized_text)
    })


@app.route("/start", methods=["POST"])
def start_camera():
    global camera_running
    camera_running = True
    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop_camera():
    global camera_running, cap
    camera_running = False
    if cap:
        cap.release()
    return jsonify({"status": "stopped"})


# ---------------- WORD PREDICTION ----------------
def predict_next_word(text):
    words = text.strip().split()
    if not words:
        return ""

    last_word = words[-1].lower()

    candidates = spell.candidates(last_word)
    if not candidates:
        return ""

    ranked = sorted(
        candidates,
        key=lambda w: zipf_frequency(w, "en"),
        reverse=True
    )

    return ranked[0]


# ---------------- SPELL CORRECTION ----------------
def correct_last_word(text):
    words = text.strip().split()
    if not words:
        return text

    last_word = words[-1]
    corrected = spell.correction(last_word)

    if corrected and corrected != last_word:
        words[-1] = corrected

    return " ".join(words) + " "


# ---------------- CAMERA ----------------
def generate_frames():
    global cap, recognized_text
    global last_prediction, stable_count
    global last_added_time, last_backspace_time

    cap = cv2.VideoCapture(0)

    while True:
        if not camera_running:
            time.sleep(0.15)
            continue

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        smoothed_prediction = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]

            data = []
            for p in lm.landmark:
                data.extend([p.x, p.y])

            data = np.array(data).reshape(1, -1)
            pred = model.predict(data)[0]

            prediction_buffer.append(pred)
            smoothed_prediction = max(
                set(prediction_buffer),
                key=prediction_buffer.count
            )

            if smoothed_prediction == last_prediction:
                stable_count += 1
            else:
                stable_count = 0
                last_prediction = smoothed_prediction

            now = time.time()

            # -------- BACKSPACE --------
            if smoothed_prediction == "BACKSPACE":
                if stable_count >= STABLE_THRESHOLD:
                    if now - last_backspace_time > BACKSPACE_REPEAT_DELAY:
                        recognized_text = recognized_text[:-1]
                        last_backspace_time = now

            # -------- SPACE --------
            elif smoothed_prediction == "SPACE":
                if stable_count >= STABLE_THRESHOLD:
                    if now - last_added_time > ADD_DELAY:
                        recognized_text = correct_last_word(recognized_text)
                        last_added_time = now
                        stable_count = 0

            # -------- LETTER --------
            elif stable_count >= STABLE_THRESHOLD:
                if now - last_added_time > ADD_DELAY:
                    recognized_text += smoothed_prediction
                    last_added_time = now
                    stable_count = 0

            cv2.putText(
                frame,
                smoothed_prediction,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(port=5000, threaded=True)