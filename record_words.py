"""Record sign clips for one word.

Usage:   python3 record_words.py hello
Keys:    SPACE = record one clip (2 s countdown, then 2 s recording)
         q     = quit
Clips are saved as small landmark files in WordData/<word>/ (no video stored).
"""
import math
import os
import sys
import time

import cv2
import numpy as np

from landmarks import (CLIP_SECONDS, SEQ_LEN, draw, extract_features,
                       make_holistic, process, resample)

if len(sys.argv) < 2:
    print('Usage: python3 record_words.py <word>   e.g. python3 record_words.py hello')
    sys.exit(1)

word = sys.argv[1].strip().lower().replace(" ", "_")
folder = os.path.join("WordData", word)
os.makedirs(folder, exist_ok=True)
count = len([f for f in os.listdir(folder) if f.endswith(".npy")])

COUNTDOWN = 2.0
cap = cv2.VideoCapture(0)
holistic = make_holistic()

state = "idle"            # idle -> countdown -> recording -> idle
t0 = 0.0
frames, hand_flags = [], []
message, message_until = "", 0.0


def put(img, text, y, scale=0.8, color=(255, 255, 255)):
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


while True:
    ok, frame = cap.read()
    if not ok:
        continue
    results = process(holistic, frame)
    draw(frame, results)
    feats, hands = extract_features(results)
    now = time.time()

    if state == "countdown":
        remaining = COUNTDOWN - (now - t0)
        if remaining <= 0:
            state, t0, frames, hand_flags = "recording", now, [], []
        else:
            put(frame, f"Get ready... {math.ceil(remaining)}", 110, 1.6, (0, 255, 255))
    elif state == "recording":
        frames.append(feats)
        hand_flags.append(hands)
        put(frame, "REC - sign now!", 110, 1.6, (0, 0, 255))
        if now - t0 >= CLIP_SECONDS:
            seen = float(np.mean(hand_flags)) if hand_flags else 0.0
            if len(frames) >= 10 and seen >= 0.5:
                clip = resample(frames, SEQ_LEN)
                np.save(os.path.join(folder, f"clip_{int(time.time() * 1000)}.npy"), clip)
                count += 1
                message = f"Saved clip #{count}"
            else:
                message = "Hands not seen enough - discarded"
            message_until = now + 1.5
            state = "idle"

    put(frame, f"Word: {word}   Clips: {count}", 35)
    if state == "idle":
        put(frame, "SPACE = record a clip   q = quit", frame.shape[0] - 15, 0.7)
    if now < message_until:
        put(frame, message, 75, 0.9, (0, 255, 0))

    cv2.imshow("Record words", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" ") and state == "idle":
        state, t0 = "countdown", now

cap.release()
cv2.destroyAllWindows()
print(f"Done. {count} clips in {folder}")
