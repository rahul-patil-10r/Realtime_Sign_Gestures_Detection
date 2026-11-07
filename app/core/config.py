from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Actions - Keep exactly the same order as training
ACTIONS = ["Hi", "How are you?", "I am fine", "Thankyou", "Great"]

# Label mapping
sequence_label = {action: idx for idx, action in enumerate(ACTIONS)}

# Model configuration
SEQUENCE_LENGTH = 60
PREDICTION_HISTORY = 10
CONFIDENCE_THRESHOLD = 0.5

# Model paths
MODEL_PATH = BASE_DIR / "saved_model" / "sign_language_model.h5"

MEDIAPIPE_MODEL_PATH = BASE_DIR / "holistic_landmarker.task"
