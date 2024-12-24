# Real-Time Emotion Detection using DeepFace and OpenCV

This Python application uses OpenCV and DeepFace to detect emotions in real-time using webcam video feed. It identifies individual faces, assigns unique IDs to them, and tracks their emotions over a specified session duration.

---

## Features

- Real-time emotion recognition using DeepFace.
- Unique identification of faces with face tracking using OpenCV.
- Saves emotion data with timestamps to an Excel file (`emotion_data.xlsx`).

---

## Requirements

The following libraries and tools are required to run the project:
- Python 3.7+
- OpenCV
- DeepFace
- pandas
- datetime

---

## How It Works

1. **Video Capture:** Captures frames from the default webcam.
2. **Face Detection:** Detects faces in the frame using OpenCV's Haar cascade classifier.
3. **Emotion Prediction:** Extracts the face region and predicts the dominant emotion using DeepFace.
4. **Tracking:** Assigns a unique ID to each detected individual.
5. **Data Storage:** Logs the detected emotion, time, and individual ID into an Excel file.

---

## Setup Instructions

1. Clone the repository or download the code:
   ```bash
   git clone https://github.com/your-username/emotion-recognition
   cd emotion-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install opencv-python deepface pandas
   ```

3. Ensure that your system has access to the `haarcascade_frontalface_default.xml` file, which is included with OpenCV.

---

## How to Run

1. Open a terminal and navigate to the project directory.

2. Run the Python script:
   ```bash
   python emotion_recognize.py
   ```

3. The application will:
   - Open a webcam feed.
   - Detect faces and recognize emotions.
   - Save emotion data to `emotion_data.xlsx`.

4. To end the session:
   - Wait for the session duration (default: 30 seconds), or
   - Press the `q` key to exit manually.

---

## Output

- **Emotion Data**: An Excel file (`emotion_data.xlsx`) containing:
  - Timestamp
  - Unique Face ID
  - Detected Emotion

- **On-Screen Visualization**: Displays webcam feed with bounding boxes and emotion labels for detected faces.

---

## Customization

- **Session Duration**: Update `session_duration` in the code to change the runtime.
- **Distance Threshold**: Modify `distance_threshold` to adjust how faces are uniquely identified.
