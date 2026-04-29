import face_recognition
import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

known_encodings = []
known_names = []

print("Starting training...")

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            
            print(f"Processing image: {path} for label: {label}")
            
            # Load the image
            image = face_recognition.load_image_file(path)
            
            # Get face encodings (128d vectors)
            # We assume one face per image for training; we take the first one [0]
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(label)
            else:
                print(f"Warning: No face found in {path}. Skipping.")

# Save the encodings and names to a pickle file
data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print(f"Training complete. Saved {len(known_encodings)} encodings to encodings.pickle.")
