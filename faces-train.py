import face_recognition
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pickle")


def find_largest_face(face_locations):
    # face_locations tuples are (top, right, bottom, left)
    # Find the largest face by area: (bottom - top) * (right - left)
    return max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))


def encode_image(path, label):
    # Load the image
    image = face_recognition.load_image_file(path)

    # Detect face locations first to find the largest face
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        print(f"Warning: No face found in {path}. Skipping.")
        return None

    largest_face = find_largest_face(face_locations)

    # Get face encodings (128d vectors) with 10 jitters for accuracy on the largest face
    encodings = face_recognition.face_encodings(image, known_face_locations=[largest_face], num_jitters=10)

    if len(encodings) == 0:
        print(f"Warning: Encoding failed for the largest face in {path}. Skipping.")
        return None

    return encodings[0]


def build_dataset(image_dir):
    known_encodings = []
    known_names = []

    for root, dirs, files in os.walk(image_dir):
        if root == image_dir:
            continue  # skip images dropped directly in images/ with no person subfolder
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()

                print(f"Processing image: {path} for label: {label}")

                encoding = encode_image(path, label)
                if encoding is not None:
                    known_encodings.append(encoding)
                    known_names.append(label)

    return known_encodings, known_names


def save_encodings(known_encodings, known_names, encodings_path):
    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_path, "wb") as f:
        pickle.dump(data, f)


def main():
    print("Starting training...")

    known_encodings, known_names = build_dataset(IMAGE_DIR)
    save_encodings(known_encodings, known_names, ENCODINGS_PATH)

    print(f"Training complete. Saved {len(known_encodings)} encodings to {ENCODINGS_PATH}.")


if __name__ == "__main__":
    main()
