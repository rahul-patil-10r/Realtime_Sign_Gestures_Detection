import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks.python.vision import HolisticLandmarker


class LandmarkService:

    def __init__(self, options, expected_features):

        self.landmarker = HolisticLandmarker.create_from_options(options)

        self.last_timestamp = -1

        # Feature layout from your training pipeline:
        # Pose = 33 * 4 = 132
        # Face = N * 3
        # Left hand = 21 * 3 = 63
        # Right hand = 21 * 3 = 63

        face_features = expected_features - 132 - 63 - 63

        if face_features < 0 or face_features % 3 != 0:
            raise ValueError(f"Unexpected model feature dimension: {expected_features}")

        self.face_count = face_features // 3

    @staticmethod
    def get_points(landmarks):

        if not landmarks:
            return []

        first = landmarks[0]

        # Flat list of NormalizedLandmark objects
        if hasattr(first, "x"):
            return landmarks

        # Nested list, such as a list of detected faces
        if len(first) > 0 and hasattr(first[0], "x"):
            return first

        return []

    @staticmethod
    def fit_features(values, expected_size):

        values = np.asarray(values, dtype=np.float32).flatten()

        if len(values) >= expected_size:
            return values[:expected_size]

        return np.pad(values, (0, expected_size - len(values)))

    def extract_keypoints(self, result):

        # Pose landmarks: 33 points * 4 values
        pose_points = self.get_points(result.pose_landmarks)

        pose = [
            [point.x, point.y, point.z, float(getattr(point, "visibility", 0.0))]
            for point in pose_points
        ]

        pose = self.fit_features(pose, 132)

        # Face landmarks: N points * 3 values
        face_points = self.get_points(result.face_landmarks)

        face = [[point.x, point.y, point.z] for point in face_points]

        face = self.fit_features(face, self.face_count * 3)

        # Left hand: 21 points * 3 values
        left_points = self.get_points(result.left_hand_landmarks)

        left_hand = [[point.x, point.y, point.z] for point in left_points]

        left_hand = self.fit_features(left_hand, 63)

        # Right hand: 21 points * 3 values
        right_points = self.get_points(result.right_hand_landmarks)

        right_hand = [[point.x, point.y, point.z] for point in right_points]

        right_hand = self.fit_features(right_hand, 63)

        return np.concatenate([pose, face, left_hand, right_hand]).astype(np.float32)

    def detect_and_extract(self, frame):

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        timestamp_ms = int(time.monotonic() * 1000)

        timestamp_ms = max(timestamp_ms, self.last_timestamp + 1)

        self.last_timestamp = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        keypoints = self.extract_keypoints(result)

        return keypoints, result

    def close(self):

        self.landmarker.close()
