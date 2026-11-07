from collections import deque

from app.core.config import ACTIONS, PREDICTION_HISTORY, CONFIDENCE_THRESHOLD


class SignRecognitionService:

    def __init__(self, landmark_service, model_service):

        self.landmark_service = landmark_service
        self.model_service = model_service

        self.sequence = deque(maxlen=model_service.sequence_length)
        self.predictions = deque(maxlen=PREDICTION_HISTORY)

        self.sentence = []

        if len(ACTIONS) != model_service.num_classes:
            raise ValueError(
                f"Expected {model_service.num_classes} actions, "
                f"but got {len(ACTIONS)}"
            )

    def process_frame(self, frame):

        # 1. Extract landmarks
        keypoints, results = self.landmark_service.detect_and_extract(frame)

        # 2. Maintain sliding window
        self.sequence.append(keypoints)

        if len(self.sequence) < self.model_service.sequence_length:
            return {
                "ready": False,
                "predicted_action": None,
                "confidence": None,
                "probabilities": None,
                "stable": False,
                "sentence": self.sentence.copy(),
            }

        # 3. Predict
        probabilities, predicted_class = self.model_service.predict(list(self.sequence))

        predicted_action = ACTIONS[predicted_class]
        confidence = float(probabilities[predicted_class])

        # 4. Maintain prediction history
        self.predictions.append(predicted_class)

        # 5. Smooth predictions
        stable_prediction = (
            len(self.predictions) == PREDICTION_HISTORY
            and len(set(self.predictions)) == 1
        )

        # 6. Apply confidence threshold
        if stable_prediction and confidence > CONFIDENCE_THRESHOLD:

            if not self.sentence or predicted_action != self.sentence[-1]:
                self.sentence.append(predicted_action)

        # 7. Keep last five recognized signs
        self.sentence = self.sentence[-5:]

        return {
            "ready": True,
            "predicted_action": predicted_action,
            "confidence": confidence,
            "stable": stable_prediction,
            "probabilities": {
                action: float(probability)
                for action, probability in zip(ACTIONS, probabilities)
            },
            "sentence": self.sentence.copy(),
        }

    def close(self):
        self.landmark_service.close()
