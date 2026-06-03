import face_recognition
import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

# Store all encodings per label to average them later
label_encodings = {}

print("Starting training...")

for root, dirs, files in os.walk(image_dir):
    if root == image_dir:
        continue  # skip images dropped directly in images/ with no person subfolder
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            
            print(f"Processing image: {path} for label: {label}")
            
            # Load the image
            image = face_recognition.load_image_file(path)
            
            # Detect face locations first to find the largest face
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) > 0:
                # Find the largest face by area: (bottom - top) * (right - left)
                largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
                
                # Get face encodings (128d vectors) with 10 jitters for accuracy on the largest face
                encodings = face_recognition.face_encodings(image, known_face_locations=[largest_face], num_jitters=10)
                
                if len(encodings) > 0:
                    if label not in label_encodings:
                        label_encodings[label] = []
                    label_encodings[label].append(encodings[0])
                else:
                    print(f"Warning: Encoding failed for the largest face in {path}. Skipping.")
            else:
                print(f"Warning: No face found in {path}. Skipping.")

known_encodings = []
known_names = []

# Average encodings per person to create a single robust signature
for label, encs in label_encodings.items():
    mean_encoding = np.mean(encs, axis=0)
    known_encodings.append(mean_encoding)
    known_names.append(label)
    print(f"Optimized signature for '{label}' averaged across {len(encs)} image(s).")

# Save the encodings and names to a pickle file
data = {"encodings": known_encodings, "names": known_names}
encodings_path = os.path.join(BASE_DIR, "encodings.pickle")
with open(encodings_path, "wb") as f:
    pickle.dump(data, f)

print(f"Training complete. Saved {len(known_encodings)} optimized encodings to {encodings_path}.")
