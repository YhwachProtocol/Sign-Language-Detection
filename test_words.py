"""Live word recognition from the webcam.

Usage: python3 test_words.py        (press q to quit)
Needs: Model/words_model.h5 and Model/words_labels.json from train_words.py
"""
import json
import time
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

from landmarks import (CLIP_SECONDS, SEQ_LEN, draw, extract_features,
                       make_holistic, process, resample)

MIN_CONF = 0.70       # show a word only above this confidence
PREDICT_EVERY = 3     # run the model every N frames
MIN_SPAN = 1.0        # seconds of history needed before predicting

model = tf.keras.models.load_model("Model/words_model.h5")
words = json.load(open("Model/words_labels.json"))

cap = cv2.VideoCapture(0)
holistic = make_holistic()

buffer = deque()                  # (timestamp, features, hands_visible)
recent = deque(maxlen=5)          # last few predictions, for smoothing
shown, shown_conf = "", 0.0
frame_no = 0


def put(img, text, y, scale=1.0, color=(255, 255, 255)):
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


while True:
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ok:
        continue
    now = time.time()
    results = process(holistic, frame)
    draw(frame, results)
    feats, hands = extract_features(results)

    buffer.append((now, feats, hands))
    while buffer and now - buffer[0][0] > CLIP_SECONDS:
        buffer.popleft()

    frame_no += 1
    span = buffer[-1][0] - buffer[0][0]
    hands_seen = np.mean([b[2] for b in buffer])
    if frame_no % PREDICT_EVERY == 0:
        if span >= MIN_SPAN and hands_seen >= 0.3:
            x = resample([b[1] for b in buffer], SEQ_LEN)[None]
            probs = model(x, training=False).numpy()[0]
            i = int(probs.argmax())
            recent.append((i, float(probs[i])))
            top, n = Counter(i for i, _ in recent).most_common(1)[0]
            confs = [c for j, c in recent if j == top]
            if n >= 3 and np.mean(confs) >= MIN_CONF:
                shown, shown_conf = words[top], float(np.mean(confs))
            else:
                shown = ""
        else:
            recent.clear()
            shown = ""

    if shown:
        put(frame, f"{shown}  {shown_conf * 100:.0f}%", 50, 1.4, (0, 255, 0))
    else:
        put(frame, "...", 50, 1.4, (200, 200, 200))

    cv2.imshow("Word recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
