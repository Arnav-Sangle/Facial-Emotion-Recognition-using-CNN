import cv2
from deepface import DeepFace
import pandas as pd
import datetime
import time

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# List to store emotion data
data = []

# Dictionary to store unique face IDs
face_ids = {}
next_face_id = 1  # Start assigning IDs from 1

# Function to calculate Euclidean distance between two points (face positions)
def calculate_distance(face1, face2):
    (x1, y1, w1, h1) = face1
    (x2, y2, w2, h2) = face2
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# Distance threshold to consider the same person (adjustable)
distance_threshold = 50

# Get the start time
start_time = time.time()

# Duration for the session (in seconds)
session_duration = 30  # The session will automatically end after 10 seconds

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to RGB format (since DeepFace works on RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    new_faces = []  # To store current frame face positions

    for (x, y, w, h) in faces:
        # Generate a unique ID for each detected face based on proximity to previous face positions
        face_position = (x, y, w, h)
        new_faces.append(face_position)

        # Find if the face is already known (within a distance threshold)
        found_face_id = None
        for known_face_position, face_id in face_ids.items():
            if calculate_distance(face_position, known_face_position) < distance_threshold:
                found_face_id = face_id
                break

        # If not found, assign a new ID
        if found_face_id is None:
            found_face_id = next_face_id
            next_face_id += 1

        # Update the known face positions with the new ones
        face_ids[face_position] = found_face_id

        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Get the current time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append the time, person ID, and emotion to the data list
        data.append([current_time, found_face_id, emotion])

        # Draw rectangle around face and label with predicted emotion and ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'ID: {found_face_id}, Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Check if 10 seconds have passed
    if time.time() - start_time > session_duration:
        break

    # Exit on 'q' press (optional if you don't want it to rely solely on 10 seconds)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['Time', 'Person_ID', 'Emotion'])

# Save the DataFrame to an Excel file
df.to_excel('emotion_data.xlsx', index=False)

print("Session ended. Emotion data saved to 'emotion_data.xlsx'")
