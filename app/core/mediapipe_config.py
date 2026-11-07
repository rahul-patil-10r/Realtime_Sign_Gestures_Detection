import mediapipe as mp

from mediapipe.tasks.python import vision

from app.core.config import MEDIAPIPE_MODEL_PATH

options = vision.HolisticLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
)
