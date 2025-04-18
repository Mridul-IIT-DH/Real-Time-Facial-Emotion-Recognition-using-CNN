import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection once for efficiency
# model_selection=0 is for short-range models (typically < 2 meters)
# min_detection_confidence=0.5 means detections with score below 50% are ignored
detector = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

def get_face(image, size=(48, 48)):
    """
    Detects the largest face in the image using MediaPipe, crops it,
    converts to grayscale, and resizes to the target `size`.

    Args:
        image (np.ndarray): The input image in BGR format (from cv2.imread or webcam).
        size (tuple): The target output size (width, height) for the face ROI.

    Returns:
        np.ndarray: The resized grayscale face ROI as a 2D numpy array,
                    or None if no face is detected or an error occurs.
    """
    # MediaPipe requires RGB images
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect faces
    results = detector.process(img_rgb)

    # Check if any faces were detected
    if not results.detections:
        # print("Debug: No face detected by MediaPipe.") # Optional debug print
        return None

    # --- Extract Bounding Box ---
    # Use the first detection (MediaPipe often returns the most prominent face first)
    detection = results.detections[0]
    bbox_relative = detection.location_data.relative_bounding_box

    # Get image dimensions
    h, w, _ = image.shape

    try:
        # Calculate absolute pixel coordinates, ensuring they are within bounds
        x1 = max(int(bbox_relative.xmin * w), 0)
        y1 = max(int(bbox_relative.ymin * h), 0)

        # Calculate width and height, ensure they are positive
        face_w = int(bbox_relative.width * w)
        face_h = int(bbox_relative.height * h)

        if face_w <= 0 or face_h <= 0:
            # print(f"Debug: Invalid bbox dimensions W={face_w}, H={face_h}") # Optional debug print
            return None # Invalid bounding box dimensions

        # Calculate bottom-right coordinates, ensuring they are within bounds
        x2 = min(x1 + face_w, w)
        y2 = min(y1 + face_h, h)

        # Final check for valid coordinate range
        if x1 >= x2 or y1 >= y2:
            # print(f"Debug: Invalid bbox coordinates x1={x1}, y1={y1}, x2={x2}, y2={y2}") # Optional debug print
            return None

        # --- Crop, Convert, and Resize ---
        # Crop the face region from the original BGR image
        face = image[y1:y2, x1:x2]

        # Check if the cropped face region is empty (can happen with edge cases)
        if face.size == 0:
            # print("Debug: Cropped face region is empty.") # Optional debug print
            return None

        # Convert the cropped face to grayscale
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale face to the target size
        face_resized = cv2.resize(face_gray, size, interpolation=cv2.INTER_AREA) # Use INTER_AREA for shrinking

        return face_resized

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"Error processing face detection bounding box or cropping: {e}")
        return None

# Example Usage (optional, for testing the function directly)
if __name__ == '__main__':
    # Load a sample image (replace with a valid path)
    try:
        sample_image = cv2.imread('./data/test/angry/PrivateTest_88305.jpg')
        if sample_image is None:
            print("Error: Could not load the sample image.")
        else:
            print("Sample image loaded.")
            # Detect and get the face
            face_roi = get_face(sample_image)

            if face_roi is not None:
                print(f"Face detected and processed. Shape: {face_roi.shape}")
                # Display the result (requires a GUI environment)
                cv2.imshow("Detected Face (Grayscale 48x48)", face_roi)
                cv2.waitKey(0) # Wait indefinitely until a key is pressed
                cv2.destroyAllWindows()
            else:
                print("No face was detected in the sample image.")
    except Exception as e:
        print(f"An error occurred during the example usage: {e}")