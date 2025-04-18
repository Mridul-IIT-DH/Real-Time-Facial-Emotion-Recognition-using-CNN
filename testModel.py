import cv2
import numpy as np
import tensorflow as tf
from utils import get_face  # Ensure this is a valid import and provides face detection
import time

# --- Configuration ---
MODEL_PATH = 'emotion_cnn_model_v2.keras'  # Make sure this matches the saved model in the same folder
LABEL_MAP_PATH = 'emotion_labels.npy'  # Ensure this file exists in the same folder
IMAGE_SIZE = (48, 48)  # Image size must match training image size
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)  # Green for the text
BOX_COLOR = (255, 0, 0)  # Blue for face box (optional)
BOX_THICKNESS = 2

# --- Load Model and Labels ---
print("Loading model and labels...")
try:
    # Load the model from the specified path
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    # Load the label map created by prepareData.py
    emotion_labels_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    emotions = [emotion_labels_map[i] for i in sorted(emotion_labels_map.keys())]
    print(f"Emotion labels loaded: {emotions}")
except Exception as e:
    print(f"Error loading model or labels: {e}")
    print("Ensure the model file 'emotion_cnn_model_v2.keras' and 'emotion_labels.npy' exist.")
    exit()

# --- Start Webcam ---
print("Starting webcam...")
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, change the index if necessary

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # --- Face Detection and Preprocessing ---
    # Use the updated get_face which returns grayscale image
    face_roi_gray = get_face(frame, size=IMAGE_SIZE)

    detected_emotion = "No Face Detected"

    if face_roi_gray is not None:
        try:
            # --- IMPORTANT: Preprocessing must match training ---
            # 1. Ensure float32 type
            face_input = face_roi_gray.astype('float32')
            # 2. Rescale (normalize)
            face_input = face_input / 255.0
            # 3. Add channel dimension (for grayscale)
            face_input = np.expand_dims(face_input, axis=-1)
            # 4. Add batch dimension
            face_input = np.expand_dims(face_input, axis=0)

            # --- Prediction ---
            preds = model.predict(face_input, verbose=0)  # Use verbose=0 to avoid print spam
            idx = np.argmax(preds[0])
            confidence = preds[0][idx] * 100

            if idx < len(emotions):  # Safety check
                detected_emotion = f"{emotions[idx]} ({confidence:.1f}%)"
            else:
                detected_emotion = "Unknown Index"

        except Exception as e:
            print(f"Error during prediction: {e}")
            detected_emotion = "Error"

    # --- Display Result ---
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - last_time)
    last_time = current_time

    # Put FPS and Emotion text on the frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, detected_emotion, (10, frame.shape[0] - 20), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-time Emotion Recognition', frame)

    # --- Quit Condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# --- Release Resources ---
cap.release()
# In Colab, cv2.destroyAllWindows() might cause issues if run directly in a cell.
# Commenting out is safer for typical Colab notebooks.
# If running locally or in an environment supporting GUIs, uncomment the next line.
# cv2.destroyAllWindows()
print("Resources released.")