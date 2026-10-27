"""Shared helpers: turn MediaPipe Holistic results into a fixed-size feature vector.

Used by record_words.py, train_words.py and test_words.py so that recording,
training and live prediction all see exactly the same kind of data.
"""
import numpy as np

SEQ_LEN = 30           # frames per sample fed to the model
CLIP_SECONDS = 2.0     # length of one recorded sign
N_POSE = 25            # upper-body pose points only (0-24)
N_FEATURES = N_POSE * 2 + 21 * 2 + 21 * 2   # x,y for pose + left hand + right hand = 134


# ---------------------------------------------------------------- MediaPipe
def make_holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )


def process(holistic, bgr_frame):
    import cv2
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return holistic.process(rgb)


def draw(bgr_frame, results):
    import mediapipe as mp
    mp_h = mp.solutions.holistic
    d = mp.solutions.drawing_utils
    d.draw_landmarks(bgr_frame, results.pose_landmarks, mp_h.POSE_CONNECTIONS)
    d.draw_landmarks(bgr_frame, results.left_hand_landmarks, mp_h.HAND_CONNECTIONS)
    d.draw_landmarks(bgr_frame, results.right_hand_landmarks, mp_h.HAND_CONNECTIONS)


# ----------------------------------------------------------------- features
def _xy(landmark_list, n):
    if landmark_list is None:
        return np.zeros((n, 2), dtype=np.float32), False
    pts = np.array([[l.x, l.y] for l in landmark_list.landmark[:n]], dtype=np.float32)
    return pts, True


def extract_features(results):
    """Returns (feature_vector[134], hands_visible).

    Points are re-centred on the middle of the shoulders and divided by the
    shoulder width, so the result does not depend on how far you sit from the
    camera or where you are in the frame. Missing body parts stay as zeros.
    """
    pose, has_pose = _xy(results.pose_landmarks, N_POSE)
    lh, has_lh = _xy(results.left_hand_landmarks, 21)
    rh, has_rh = _xy(results.right_hand_landmarks, 21)

    if has_pose:
        origin = (pose[11] + pose[12]) / 2.0
        scale = float(np.linalg.norm(pose[11] - pose[12]))
        if scale < 1e-3:
            scale = 0.3
    else:
        origin = np.array([0.5, 0.5], dtype=np.float32)
        scale = 0.3

    def norm(p, present):
        return (p - origin) / scale if present else p

    feats = np.concatenate([
        norm(pose, has_pose).ravel(),
        norm(lh, has_lh).ravel(),
        norm(rh, has_rh).ravel(),
    ]).astype(np.float32)
    return feats, (has_lh or has_rh)


def resample(frames, n=SEQ_LEN):
    """Pick n evenly spaced frames, so clips recorded at any camera fps match."""
    frames = np.asarray(frames, dtype=np.float32)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]
