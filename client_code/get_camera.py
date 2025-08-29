import cv2
import argparse

def get_camera(server_url):
    cap = cv2.VideoCapture(server_url)

    if not cap.isOpened():
        print(f"Failed to connect to stream at {server_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam Stream", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
