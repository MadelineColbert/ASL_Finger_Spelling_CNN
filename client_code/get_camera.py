import cv2
import argparse
from client_code.model_handling import preprocess_image, get_prediction

def get_camera(server_url):
    cap = cv2.VideoCapture(server_url)

    if not cap.isOpened():
        print(f"Failed to connect to stream at {server_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ml_frame = preprocess_image(frame)
        letter_prediction = get_prediction(ml_frame)
        cv2.imshow("Webcam Stream", frame)
        print(letter_prediction)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
