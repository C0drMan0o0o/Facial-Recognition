# Face Recognition (Deep Learning)

Recognizes known faces in a live webcam feed using modern Deep Learning (dlib's ResNet model via the `face_recognition` library). This system is significantly more accurate than traditional Haar Cascades and LBPH methods.

## Features
- **High Accuracy:** Uses a 128-dimension face encoding for robust recognition.
- **Lighting Normalization (CLAHE):** Integrates Contrast Limited Adaptive Histogram Equalization on the Y-luminance channel to handle shadows and uneven lighting.
- **Alternate Frame Processing:** Performs detection and recognition on every other frame to minimize CPU usage and maintain high FPS.
- **Robust Training:** Detects face locations to filter and train only on the largest face (the primary subject) in images containing multiple people, storing each image's encoding individually so multiple photos per person improve match coverage.
- **Dynamic Visual HUD:** Displays green bounding boxes for recognized users with confidence percentages (e.g., `Sanjith (88%)`), and red boxes for unknown individuals.

## Tech Stack
- **Python 3**
- **face_recognition** (Deep Learning engine)
- **OpenCV** (Video capture and display)
- **NumPy** (Mathematical operations)

## How to Run

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset:**
   Place training images in the `images/` folder. Create a separate subfolder for each person (the folder name will be used as their label).

4. **Train the Model:**
   Extract face encodings from your images:
   ```bash
   python faces-train.py
   ```
   This generates `encodings.pickle`.

5. **Run Recognition:**
   Start the real-time webcam recognition:
   ```bash
   python "Face Recognition.py"
   ```
   Press **'q'** to quit the webcam window.

## Troubleshooting
- **Camera Access (macOS):** Ensure your Terminal or IDE has permission to access the camera in *System Settings > Privacy & Security > Camera*.
- **Camera Access (Windows):** Ensure "Let desktop apps access your camera" is turned ON in *Settings > Privacy & Security > Camera*.
- **Performance:** The script resizes the internal processing frame to 1/2 size for high-resolution detection and uses alternate-frame rendering. If it's still slow, ensure you aren't running other CPU-heavy applications in the background.
