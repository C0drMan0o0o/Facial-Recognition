import face_recognition
import cv2
import numpy as np
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")
TOLERANCE = 0.5


def load_encodings(encodings_path):
    if not os.path.exists(encodings_path):
        print("Error: encodings.pickle not found. Please run faces-train.py first.")
        sys.exit(1)

    with open(encodings_path, 'rb') as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    if len(known_encodings) == 0:
        print("Error: No encodings found in encodings.pickle. Please add images and re-run faces-train.py.")
        sys.exit(1)

    return known_encodings, known_names


def open_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    return cap


def normalize_lighting(small_frame, clahe):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on Y-channel of YUV
    # to normalize lighting/shadows while preserving colors.
    yuv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])

    # Convert normalized image from YUV to RGB color (face_recognition)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def match_face(face_encoding, known_encodings, known_names):
    name = "Unknown"
    confidence = 0

    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] <= TOLERANCE:
            name = known_names[best_match_index]
            distance = face_distances[best_match_index]
            confidence = min(100, int((1.0 - distance / TOLERANCE) * 100))

    return name, confidence


def detect_faces(frame, known_encodings, known_names, clahe):
    # Resize frame to 1/2 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    rgb_small_frame = normalize_lighting(small_frame, clahe)

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Scale to full-frame coordinates here so the display loop uses
    # the same locations on both processing and cached frames.
    face_locations = [(top * 2, right * 2, bottom * 2, left * 2)
                      for (top, right, bottom, left) in face_locations]

    face_names = []
    face_confidences = []
    for face_encoding in face_encodings:
        name, confidence = match_face(face_encoding, known_encodings, known_names)
        face_names.append(name)
        face_confidences.append(confidence)

    return face_locations, face_names, face_confidences


def draw_results(frame, face_locations, face_names, face_confidences):
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
        # Green for known, Red for unknown
        if name == "Unknown":
            color = (0, 0, 255)  # BGR Red
            display_text = "Unknown"
        else:
            color = (0, 255, 0)  # BGR Green
            display_text = f"{name} ({confidence}%)"

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, display_text, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)


def run_recognition_loop(cap, known_encodings, known_names):
    # Create CLAHE once (reused every frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Initialize variables for alternate frame processing
    process_this_frame = True
    face_locations = []
    face_names = []
    face_confidences = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally (mirror view) to prevent it from being inverted
        frame = cv2.flip(frame, 1)

        # Process alternate frames to save CPU / boost FPS
        if process_this_frame:
            face_locations, face_names, face_confidences = detect_faces(
                frame, known_encodings, known_names, clahe)

        process_this_frame = not process_this_frame

        # Display the results
        draw_results(frame, face_locations, face_names, face_confidences)

        # Display the resulting image
        cv2.imshow('Face Recognition', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    print("Loading encodings...")
    known_encodings, known_names = load_encodings(ENCODINGS_PATH)

    cap = open_webcam()
    print("Starting webcam... Press 'q' to quit.")

    try:
        run_recognition_loop(cap, known_encodings, known_names)
    finally:
        # Release handle to the webcam
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
