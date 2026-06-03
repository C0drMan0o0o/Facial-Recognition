import face_recognition
import cv2
import numpy as np
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")
TOLERANCE = 0.5

# Load the known faces and encodings
print("Loading encodings...")
if not os.path.exists(ENCODINGS_PATH):
    print("Error: encodings.pickle not found. Please run faces-train.py first.")
    sys.exit(1)

with open(ENCODINGS_PATH, 'rb') as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

print("Starting webcam... Press 'q' to quit.")

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
        # Resize frame to 1/2 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on Y-channel of YUV
        # to normalize lighting/shadows while preserving colors.
        yuv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        
        # Convert normalized image from YUV to RGB color (face_recognition)
        rgb_small_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_confidences = []
        for face_encoding in face_encodings:
            # Compare face with lower tolerance for higher accuracy
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"
            confidence = 0

            if len(known_encodings) > 0:
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    # Compute confidence: 0.0 distance = 100%, at tolerance threshold = 0%
                    distance = face_distances[best_match_index]
                    confidence = max(0, min(100, int((1.0 - distance / TOLERANCE) * 100)))

            face_names.append(name)
            face_confidences.append(confidence)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
        # Scale back up face locations since the frame we detected in was scaled to 1/2 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

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



    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
