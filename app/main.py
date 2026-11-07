import cv2
import numpy as np

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.core.config import MODEL_PATH
from app.core.mediapipe_config import options

from app.services.model_service import ModelService
from app.services.landmark_service import LandmarkService
from app.services.sign_recognition_service import SignRecognitionService

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"


# Load model once during application startup
@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model_service = ModelService(MODEL_PATH)

    print("SIGNSENSE model loaded successfully")

    yield

    print("SIGNSENSE application shutting down")


app = FastAPI(
    title="SIGNSENSE API",
    description="Real-time Sign Language Recognition",
    lifespan=lifespan,
)


# Serve frontend
@app.get("/")
async def home():
    return FileResponse(FRONTEND_PATH)


# Health check
@app.get("/health")
async def health():
    return {"status": "running", "application": "SIGNSENSE"}


# WebSocket endpoint
@app.websocket("/ws/sign/")
async def sign_recognition(websocket: WebSocket):

    await websocket.accept()

    landmark_service = None
    recognition_service = None

    model_service = websocket.app.state.model_service

    try:

        # Initialize client-specific services
        landmark_service = LandmarkService(options, model_service.num_features)

        recognition_service = SignRecognitionService(landmark_service, model_service)

        print("WebSocket client connected")

        while True:

            # Receive JPEG bytes
            frame_bytes = await websocket.receive_bytes()

            # Decode image
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)

            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Invalid image frame"})
                continue

            # Keep this flip only if frontend sends unmirrored frames
            frame = cv2.flip(frame, 1)

            # Recognition
            result = recognition_service.process_frame(frame)

            # Send response
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")

    except Exception as error:

        print(f"WebSocket error: {error}")

        try:
            await websocket.close(code=1011)
        except Exception:
            pass

    finally:

        if recognition_service is not None:
            recognition_service.close()

        elif landmark_service is not None:
            landmark_service.close()
