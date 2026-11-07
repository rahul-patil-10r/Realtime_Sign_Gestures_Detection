import asyncio
import websockets
import cv2
import json


async def main():

    # Open laptop webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access webcam")
        return

    uri = "ws://127.0.0.1:8000/ws/sign/"

    try:
        async with websockets.connect(uri) as websocket:

            print("WebSocket connected!")

            while True:

                ret, frame = cap.read()

                if not ret:
                    print("Failed to capture frame")
                    break

                # Mirror webcam
                frame = cv2.flip(frame, 1)

                # Convert frame to JPEG bytes
                success, buffer = cv2.imencode(".jpg", frame)

                if not success:
                    continue

                # Send frame to backend
                await websocket.send(buffer.tobytes())

                # Receive prediction
                response = await websocket.recv()
                result = json.loads(response)

                print(result)

                # Display webcam
                cv2.imshow("SIGNSENSE", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")

    finally:
        cap.release()
        cv2.destroyAllWindows()


asyncio.run(main())
