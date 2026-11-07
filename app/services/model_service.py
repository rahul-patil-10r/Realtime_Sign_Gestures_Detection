import numpy as np
from tensorflow.keras.models import load_model


class ModelService:

    def __init__(self, model_path="saved_model/sign_language_model.h5"):

        self.model = load_model(model_path)

        self.sequence_length = self.model.input_shape[1]
        self.num_features = self.model.input_shape[2]
        self.num_classes = self.model.output_shape[-1]

        print("LSTM model loaded successfully")
        print("Sequence length:", self.sequence_length)
        print("Features:", self.num_features)
        print("Classes:", self.num_classes)

    def predict(self, sequence):

        sequence = np.asarray(sequence, dtype=np.float32)

        expected_shape = (self.sequence_length, self.num_features)

        if sequence.shape != expected_shape:
            raise ValueError(f"Expected {expected_shape}, got {sequence.shape}")

        if not np.isfinite(sequence).all():
            raise ValueError("Input contains NaN or infinite values")

        batch = np.expand_dims(sequence, axis=0)

        probabilities = self.model.predict(batch, verbose=0)[0]

        predicted_class = int(np.argmax(probabilities))

        return probabilities, predicted_class
