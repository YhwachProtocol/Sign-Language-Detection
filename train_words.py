"""Train the word model from the clips in WordData/.

Usage:  python3 train_words.py
Output: Model/words_model.h5 and Model/words_labels.json
"""
import json
import os

import numpy as np
import tensorflow as tf

from landmarks import N_FEATURES, SEQ_LEN

DATA_DIR = "WordData"
MODEL_PATH = "Model/words_model.h5"
LABELS_PATH = "Model/words_labels.json"
EPOCHS = 150
AUG_COPIES = 6        # extra randomised copies of every training clip
rng = np.random.default_rng(42)


def load_data():
    words = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and any(f.endswith(".npy") for f in os.listdir(os.path.join(DATA_DIR, d)))
    )
    if len(words) < 2:
        raise SystemExit("Need clips for at least 2 words in WordData/. Run record_words.py first.")
    X, y = [], []
    for i, w in enumerate(words):
        for f in sorted(os.listdir(os.path.join(DATA_DIR, w))):
            if f.endswith(".npy"):
                clip = np.load(os.path.join(DATA_DIR, w, f))
                if clip.shape == (SEQ_LEN, N_FEATURES):
                    X.append(clip)
                    y.append(i)
    return words, np.array(X, dtype=np.float32), np.array(y)


def augment(x):
    """Random shift in time, size, position and a little noise."""
    k = int(rng.integers(-4, 5))
    if k > 0:
        x = np.concatenate([np.repeat(x[:1], k, axis=0), x[:-k]])
    elif k < 0:
        x = np.concatenate([x[-k:], np.repeat(x[-1:], -k, axis=0)])
    pts = x.reshape(SEQ_LEN, -1, 2)
    present = (pts != 0).any(axis=-1, keepdims=True)     # keep "missing" points at zero
    scale = rng.uniform(0.9, 1.1)
    shift = rng.uniform(-0.1, 0.1, size=(1, 1, 2))
    noise = rng.normal(0, 0.01, pts.shape)
    pts = np.where(present, pts * scale + shift + noise, 0.0)
    return pts.reshape(SEQ_LEN, -1).astype(np.float32)


def split(y, val_fraction=0.2):
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) >= 3 else 0
        val_idx += list(idx[:n_val])
        train_idx += list(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


words, X, y = load_data()
print(f"\nWords ({len(words)}): {words}")
for i, w in enumerate(words):
    print(f"  {w}: {int((y == i).sum())} clips")
few = [w for i, w in enumerate(words) if (y == i).sum() < 10]
if few:
    print(f"\nWARNING: fewer than 10 clips for {few}. Aim for 20-30 per word.\n")

tr, va = split(y)
X_train, y_train = X[tr], y[tr]
X_val, y_val = X[va], y[va]

X_aug = [augment(x) for x in X_train for _ in range(AUG_COPIES)]
y_aug = [c for c in y_train for _ in range(AUG_COPIES)]
X_train = np.concatenate([X_train, np.array(X_aug)])
y_train = np.concatenate([y_train, np.array(y_aug)])
order = rng.permutation(len(X_train))
X_train, y_train = X_train[order], y_train[order]
print(f"Training samples (with augmentation): {len(X_train)}   Validation clips: {len(X_val)}")

L = tf.keras.layers
model = tf.keras.Sequential([
    L.Input((SEQ_LEN, N_FEATURES)),
    L.Conv1D(64, 3, padding="same", activation="relu"),
    L.LSTM(64, return_sequences=True),
    L.LSTM(64),
    L.Dropout(0.3),
    L.Dense(64, activation="relu"),
    L.Dense(len(words), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [tf.keras.callbacks.EarlyStopping(
    monitor="val_loss" if len(X_val) else "loss", patience=20, restore_best_weights=True)]
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val) if len(X_val) else None,
    epochs=EPOCHS, batch_size=16, callbacks=callbacks, verbose=2,
)

if len(X_val):
    pred = model.predict(X_val, verbose=0).argmax(axis=1)
    print(f"\nValidation accuracy: {(pred == y_val).mean() * 100:.1f}%")
    print("Per word:")
    for i, w in enumerate(words):
        m = y_val == i
        if m.any():
            print(f"  {w}: {(pred[m] == i).mean() * 100:.0f}%  ({int(m.sum())} clips)")
    confused = {}
    for t, p in zip(y_val, pred):
        if t != p:
            confused[(words[t], words[p])] = confused.get((words[t], words[p]), 0) + 1
    if confused:
        print("Most confused (true -> predicted):")
        for (a, b), n in sorted(confused.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {a} -> {b}: {n}")

os.makedirs("Model", exist_ok=True)
model.save(MODEL_PATH)
with open(LABELS_PATH, "w") as f:
    json.dump(words, f)
print(f"\nSaved {MODEL_PATH} and {LABELS_PATH}")
