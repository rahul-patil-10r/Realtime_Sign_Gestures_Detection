# SIGNSENSE — Real-Time Sign Gesture Recognition

SIGNSENSE is a real-time, webcam-based gesture recognition application. It uses **MediaPipe Holistic Landmarker** to extract body, face, and hand landmarks, and an **LSTM neural network** to classify a sequence of frames. A **FastAPI WebSocket backend** streams predictions to a browser-based frontend.

> **Note:** The five labels below are the custom gestures used to collect and train this project’s dataset. They should not be treated as authoritative definitions of standardized sign language.

## Features

- Live webcam input through a browser.
- Body, face, and hand landmark extraction with MediaPipe Tasks.
- Sequence-based gesture classification using an LSTM model.
- Real-time predictions delivered through FastAPI WebSockets.
- Prediction confidence and recognized-gesture sentence display.
- Five custom gesture classes.

## Supported gestures

| Model label | Gesture used in this project |
|---|---|
| `Hi` | Raise one hand and wave toward the camera. |
| `How are you?` | Turn the hand upside down and wave horizontally. |
| `I am fine` | Make a thumbs-up gesture. |
| `Thankyou` | Perform the hand-to-forehead gesture used in the training recordings. |
| `Great` | Perform a salute. |

For reliable predictions, perform the gestures in a similar way to the examples used during data collection. Keep your hands and upper body visible in the camera.

## Technology stack

- **Python**
- **TensorFlow / Keras** — LSTM model training and inference
- **MediaPipe Tasks** — Holistic landmark detection
- **OpenCV** — frame decoding and image processing
- **NumPy** — feature arrays and model input preparation
- **FastAPI** — backend API
- **WebSocket** — continuous browser-to-server frame streaming
- **HTML, CSS, JavaScript** — frontend

## Model and input details

The current model was trained with the following dataset configuration:

| Property | Value |
|---|---:|
| Gesture classes | 5 |
| Sequences per class | 60 |
| Total sequences | 300 |
| Frames per sequence | 60 |
| Features per frame | 1,692 |
| Model input shape | `(60, 1692)` |
| Model output classes | 5 |

Each frame contains:

- Pose: 33 landmarks × 4 values (`x`, `y`, `z`, visibility) = 132 features
- Face: 478 landmarks × 3 values (`x`, `y`, `z`) = 1,434 features
- Left hand: 21 landmarks × 3 values = 63 features
- Right hand: 21 landmarks × 3 values = 63 features

**Total: 132 + 1,434 + 63 + 63 = 1,692 features per frame.**

The model architecture is:

```text
Input: (60, 1692)
    |
LSTM(64, return_sequences=True)
    |
Dropout(0.2)
    |
LSTM(128, return_sequences=True)
    |
Dropout(0.2)
    |
LSTM(64)
    |
Dense(64, ReLU)
    |
Dense(32, ReLU)
    |
Dense(5, Softmax)
```

The model uses the Adam optimizer and categorical cross-entropy loss for one-hot encoded labels.

> Training accuracy alone does not establish performance on unseen users or recording conditions. Evaluate the model on separate recordings that were not used during training before making accuracy claims.

## Project structure

The following is the intended high-level structure. Adjust notebook and test filenames to match the files in your repository.

```text
SIGNSENSE/
├── app/
│   ├── core/
│   │   ├── config.py
│   │   └── mediapipe_config.py
│   ├── services/
│   │   ├── model_service.py
│   │   ├── landmark_service.py
│   │   └── sign_recognition_service.py
│   └── main.py
├── frontend/
│   └── index.html
├── saved_model/
│   └── sign_language_model.h5
├── notebooks/
│   ├── data_collection.ipynb
│   ├── model_training.ipynb
│   └── inference.ipynb
├── tests/
├── holistic_landmarker.task
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Replace the URL and directory with your actual GitHub repository details.

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python version compatible with the dependency versions in `requirements.txt`. If you encounter an installation issue, check the TensorFlow and MediaPipe compatibility for your operating system and Python version.

### 4. Check required model assets

Make sure these files exist at the paths expected by `app/core/config.py`:

```text
saved_model/sign_language_model.h5
holistic_landmarker.task
```

The trained Keras model and the MediaPipe task model are separate assets; both are needed for inference.

## Run the application

From the repository root, start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open the frontend in your browser:

```text
http://127.0.0.1:8000/
```

Allow camera access when the browser asks. Select **Start Recognition**, perform one of the five gestures in view of the camera, and watch the detected label, confidence, and recognized sentence.

Select **Stop** to stop recognition and release the webcam.

## Application endpoints

| Endpoint | Type | Purpose |
|---|---|---|
| `/` | HTTP GET | Serves the browser frontend. |
| `/health` | HTTP GET | Returns application health status. |
| `/ws/sign/` | WebSocket | Receives webcam frames and sends recognition results. |

The frontend sends JPEG frames to `/ws/sign/`. The backend decodes each frame, extracts landmarks, maintains a rolling sequence of 60 frames, runs model inference, and returns the prediction data.

## Dataset and training

The training dataset contains 300 sequences across five custom gesture classes. Each sequence contains 60 frames, and each frame is represented by 1,692 landmark features.

If you publish the dataset, make sure it contains no private or identifiable recordings without the appropriate consent. The dataset is not required to run inference if the trained model and MediaPipe task asset are available.

## Troubleshooting

- **Camera permission error:** Allow camera access for the browser and ensure another application is not exclusively using the camera.
- **WebSocket connection error:** Confirm the FastAPI server is running and that the frontend connects to `/ws/sign/` with the correct `ws://` or `wss://` protocol.
- **Model file not found:** Verify `MODEL_PATH` in `app/core/config.py` and confirm the model file exists.
- **MediaPipe task file not found:** Verify `MEDIAPIPE_MODEL_PATH` and confirm `holistic_landmarker.task` exists.
- **Unexpected predictions:** Check that the training and backend landmark extraction use the same landmark order, feature dimensions, frame orientation, and preprocessing. Confirm that the backend action order exactly matches the class order used during training.

## Limitations and future improvements

- The current classifier recognizes five project-specific gestures.
- Recognition quality can vary with lighting, camera position, distance, and gesture execution.
- The current dataset is relatively small; collecting more varied recordings and evaluating on held-out sessions can improve confidence in generalization.
- Future work may include a larger gesture vocabulary, more robust temporal smoothing, formal evaluation metrics, and deployment optimization.

## License

Add a license before permitting reuse or redistribution. Until a license is added, do not assume that others have permission to reuse this project's code or data.
