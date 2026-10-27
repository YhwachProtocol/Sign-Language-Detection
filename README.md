# Sign Language Detection

Real-time sign language to text translation using a webcam. The app detects your hand, crops it onto a white background, and classifies the sign with a Keras model trained on Google Teachable Machine. Results are shown in a Flask web interface.

## Features

- Live hand detection and tracking (cvzone / MediaPipe)
- Mirrored camera view for natural signing
- Sign classification with a Keras (`.h5`) model
- Web interface with live prediction, confidence bar and transcript
- Scripts for collecting your own training images

## Project structure

| File / folder | Purpose |
|---|---|
| `app.py` | Flask app: camera feed, prediction, web interface |
| `datacollection.py` | Collects hand images for one class (press `s` to save) |
| `record_words.py`, `train_words.py`, `test_words.py` | Word-level recording, training and testing |
| `landmarks.py`, `hand_landmarker.task` | Hand landmark detection |
| `Model/` | `keras_model.h5` and `labels.txt` exported from Teachable Machine |
| `templates/index.html` | Front end |

## Setup

```bash
git clone https://github.com/YhwachProtocol/Sign-Language-Detection.git
cd Sign-Language-Detection
python3 -m venv venv
source venv/bin/activate
pip install opencv-python cvzone mediapipe numpy flask "tensorflow<2.16" pillow
```

## Run

```bash
python app.py
```

Open the local URL printed in the terminal (usually http://127.0.0.1:5000) and allow camera access.

## Train on your own signs

1. Open `datacollection.py` and set the class folder, for example `folder = "Letters/A"`.
2. Run `python datacollection.py`, show the sign inside the box and press `s` to save images. Collect about 200-300 images per sign, moving your hand slightly between saves.
3. Repeat for every sign you want to recognize.
4. Create a **Standard image model** on [Teachable Machine](https://teachablemachine.withgoogle.com), add one class per sign, upload the images and click **Train Model**.
5. Export as **Tensorflow > Keras**, then replace the contents of the `Model/` folder with the downloaded `keras_model.h5` and `labels.txt`.
6. Run `python app.py` again.

Tip: add a "Nothing" class (relaxed hand or no hand) so the model does not guess a sign when you are not signing.

## Notes

- The camera frame is flipped horizontally in the data collection and app scripts. Keep this consistent between training and testing.
- Training images (`Letters/`, `WordData/`, `Data/`) are not included in the repo.

## Status

Work in progress. Currently trained on the letters A, B and C, with the rest of the alphabet and words to come.
