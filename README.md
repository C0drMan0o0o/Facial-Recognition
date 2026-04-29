# Face Recognition (Deep Learning)

Recognizes known faces in a live webcam feed using modern Deep Learning (dlib's ResNet model via the `face_recognition` library). This system is significantly more accurate than traditional Haar Cascades and LBPH methods.

## Features
- **High Accuracy:** Uses a 128-dimension face encoding for robust recognition.
- **Real-time Optimization:** Frames are resized for faster processing while maintaining high-quality display.
- **Easy Training:** Simply add images to the `images/` folder and run the training script.

## Tech Stack
- **Python 3**
- **face_recognition** (Deep Learning engine)
- **OpenCV** (Video capture and display)
- **NumPy** (Mathematical operations)

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset:**
   Place training images in the `images/` folder. Create a separate subfolder for each person (the folder name will be used as their label).

3. **Train the Model:**
   Extract face encodings from your images:
   ```bash
   python faces-train.py
   ```
   This generates `encodings.pickle`.

4. **Run Recognition:**
   Start the real-time webcam recognition:
   ```bash
   python "Face Recognition.py"
   ```
   Press **'q'** to quit the webcam window.

## Troubleshooting
- **Camera Access (macOS):** Ensure your Terminal or IDE has permission to access the camera in *System Settings > Privacy & Security > Camera*.
- **Camera Access (Windows):** Ensure "Let desktop apps access your camera" is turned ON in *Settings > Privacy & Security > Camera*.
- **Performance:** The script resizes the internal processing frame to 1/4 size for speed. If it's still slow, ensure you aren't running other heavy applications in the background.
