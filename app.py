import math
import threading
import time
import webbrowser
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

MODEL_PATH = "Model/keras_model.h5"
LABELS_PATH = "Model/labels.txt"

offset = 20
imgSize = 300
insetSize = 150
HOLD_SECONDS = 1.2   # hold a sign this long to add it to the word
MIN_CONF = 0.80      # ignore predictions below this confidence

# The four pipeline stages timed for every frame that has a hand in view.
#   capture     - reading a frame from the webcam
#   preprocess  - cropping the hand and resizing it to 300x300 for the model
#   inference   - hand tracking (MediaPipe) + sign classification (CNN)
#   display     - drawing the overlays and encoding the frame for the browser
STAGES = ("capture", "preprocess", "inference", "display")
RECENT_FRAMES = 30    # window for the "live" numbers
HISTORY_FRAMES = 60   # points in the sparkline


def load_labels(path):
    """Reads Teachable Machine's labels.txt ("0 Hello", "1 No", ...)."""
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return labels


labels = load_labels(LABELS_PATH)
detector = HandDetector(maxHands=1)
classifier = Classifier(MODEL_PATH, LABELS_PATH)

lock = threading.Lock()
latest_jpeg = None
state = {"label": "", "confidence": 0.0, "hand": False, "progress": 0.0,
         "word": "", "camera": True}

timing_sum = dict.fromkeys(STAGES, 0.0)
timing_frames = 0
timing_recent = deque(maxlen=HISTORY_FRAMES)   # one {stage: ms} dict per frame


def ms_since(t0):
    return (time.perf_counter() - t0) * 1000.0


def record_timing(frame_ms):
    """Call with the lock held."""
    global timing_frames
    for s in STAGES:
        timing_sum[s] += frame_ms[s]
    timing_frames += 1
    timing_recent.append(frame_ms)


def reset_timing():
    """Call with the lock held."""
    global timing_frames
    for s in STAGES:
        timing_sum[s] = 0.0
    timing_frames = 0
    timing_recent.clear()


def with_total(avg):
    total = sum(avg[s] for s in STAGES)
    out = dict(avg)
    out["total"] = total
    out["fps"] = 1000.0 / total if total > 0 else 0.0
    return out


def timing_snapshot():
    """Call with the lock held."""
    n = timing_frames
    avg = {s: (timing_sum[s] / n if n else 0.0) for s in STAGES}
    recent = list(timing_recent)[-RECENT_FRAMES:]
    live = {s: (sum(f[s] for f in recent) / len(recent) if recent else 0.0) for s in STAGES}
    return {
        "frames": n,
        "avg": with_total(avg),
        "live": with_total(live),
        "history": [round(sum(f.values()), 2) for f in timing_recent],
    }


def camera_loop():
    global latest_jpeg
    cap = cv2.VideoCapture(0)
    with lock:
        state["camera"] = bool(cap.isOpened())
    candidate, candidate_start, added = None, 0.0, False

    while True:
        # --- Stage 1: frame capture ---------------------------------------
        t0 = time.perf_counter()
        success, img = cap.read()
        img = cv2.flip(img, 1)
        capture_ms = ms_since(t0)
        if not success:
            time.sleep(0.05)
            continue

        # --- hand tracking (counted under "inference") --------------------
        t0 = time.perf_counter()
        hands, img = detector.findHands(img)
        detect_ms = ms_since(t0)

        t0 = time.perf_counter()
        imgOutput = img.copy()
        display_ms = ms_since(t0)

        label, conf, progress = "", 0.0, 0.0
        predicted = False
        preprocess_ms = classify_ms = 0.0

        if hands:
            x, y, w, h = hands[0]['bbox']
            imgH, imgW = img.shape[:2]
            y1, y2 = max(0, y - offset), min(imgH, y + h + offset)
            x1, x2 = max(0, x - offset), min(imgW, x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size > 0 and w > 0 and h > 0:
                # --- Stage 2: preprocessing -------------------------------
                t0 = time.perf_counter()
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                if h / w > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                preprocess_ms = ms_since(t0)

                # --- Stage 3: classification (hand tracking was timed above)
                t0 = time.perf_counter()
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                classify_ms = ms_since(t0)
                predicted = True
                label = labels[index] if index < len(labels) else str(index)
                conf = float(prediction[index])

                # --- Stage 4 (part): drawing overlays ---------------------
                t0 = time.perf_counter()
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 0), 3)
                inset = cv2.resize(imgWhite, (insetSize, insetSize))
                outH, outW = imgOutput.shape[:2]
                if outH > insetSize + 20 and outW > insetSize + 20:
                    top, left = 10, outW - insetSize - 10
                    imgOutput[top:top + insetSize, left:left + insetSize] = inset
                    cv2.rectangle(imgOutput, (left, top),
                                  (left + insetSize, top + insetSize), (0, 255, 0), 2)
                display_ms += ms_since(t0)

        # Hold-to-add logic
        now = time.time()
        with lock:
            if label and conf >= MIN_CONF:
                if label == candidate:
                    elapsed = now - candidate_start
                    progress = min(1.0, elapsed / HOLD_SECONDS)
                    if elapsed >= HOLD_SECONDS and not added:
                        state["word"] += label if len(label) == 1 else label + " "
                        added = True
                else:
                    candidate, candidate_start, added = label, now, False
            else:
                candidate, added = None, False

            state.update({"label": label, "confidence": conf,
                          "hand": bool(hands), "progress": progress})

        # --- Stage 4 (part): encoding the frame for the browser -----------
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", imgOutput, [cv2.IMWRITE_JPEG_QUALITY, 80])
        display_ms += ms_since(t0)

        with lock:
            if ok:
                latest_jpeg = buf.tobytes()
            if predicted:
                record_timing({
                    "capture": capture_ms,
                    "preprocess": preprocess_ms,
                    "inference": detect_ms + classify_ms,
                    "display": display_ms,
                })


app = Flask(__name__)


def mjpeg():
    while True:
        with lock:
            frame = latest_jpeg
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html", classes=labels)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def get_state():
    with lock:
        data = dict(state)
        data["timing"] = timing_snapshot()
        return jsonify(data)


@app.route("/timing.csv")
def timing_csv():
    with lock:
        snap = timing_snapshot()
    names = {"capture": "Frame capture", "preprocess": "Preprocessing",
             "inference": "Feature extraction & classification", "display": "Display"}
    rows = ["stage,average_ms"]
    rows += [f'"{names[s]}",{snap["avg"][s]:.3f}' for s in STAGES]
    rows += [f'"Total",{snap["avg"]["total"]:.3f}',
             f'"Frames per second",{snap["avg"]["fps"]:.2f}',
             f'"Frames measured",{snap["frames"]}']
    return Response("\n".join(rows) + "\n", mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=timing.csv"})


@app.route("/action", methods=["POST"])
def action():
    act = (request.get_json(silent=True) or {}).get("action")
    with lock:
        if act == "clear":
            state["word"] = ""
        elif act == "space":
            state["word"] += " "
        elif act == "backspace":
            state["word"] = state["word"][:-1]
        elif act == "reset_timing":
            reset_timing()
        return jsonify({"word": state["word"]})


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    url = "http://127.0.0.1:5001"
    print(f"\nOpen {url} in your browser\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)