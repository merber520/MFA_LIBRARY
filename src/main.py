import cv2
import os
import dlib
import numpy as np
import face_recognition
import argparse

image1 = 'image1.png'
image2 = 'image2.png'
#PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

def detect_face_haar(image_path1, image_path2):
    try:
        # Load classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Process first image
        image1 = cv2.imread(image_path1)
        if image1 is None:
            raise ValueError(f"Cannot load image: {image_path1}")
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        faces1 = face_cascade.detectMultiScale(
            gray1,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces1) == 0:
            print("No faces detected in first image")
            face1 = None
        elif len(faces1) > 1:
            print("Multiple faces detected in first image")
            face1 = None
        else:
            face1 = faces1[0]

        # Process second image
        image2 = cv2.imread(image_path2)
        if image2 is None:
            raise ValueError(f"Cannot load image: {image_path2}")
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        faces2 = face_cascade.detectMultiScale(
            gray2,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces2) == 0:
            print("No faces detected in second image")
            face2 = None
        elif len(faces2) > 1:
            print("Multiple faces detected in second image")
            face2 = None
        else:
            face2 = faces2[0]

        return face1, face2

    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return None, None

def get_face_encodings(image_path1, face1, image_path2, face2):
    """
    Extract 128-dimensional face encodings for two faces.
    
    Args:
        image_path1 (str): Path to the first image.
        face1 (tuple): Coordinates of the first face (x, y, w, h).
        image_path2 (str): Path to the second image.
        face2 (tuple): Coordinates of the second face (x, y, w, h).
    
    Returns:
        tuple: (encoding1, encoding2) - Encodings for the two faces.
               Returns None for any face where encoding cannot be generated.
    """
    # Encoding for the first face
    image1 = face_recognition.load_image_file(image_path1)
    encoding1 = None
    if face1 is not None:
        face_locations1 = [(face1[1], face1[0] + face1[2], face1[1] + face1[3], face1[0])]
        encodings1 = face_recognition.face_encodings(image1, face_locations1)
        if len(encodings1) == 0:
            print("No encoding generated for face1.")
        else:
            encoding1 = encodings1[0]
    
    # Encoding for the second face
    image2 = face_recognition.load_image_file(image_path2)
    encoding2 = None
    if face2 is not None:
        face_locations2 = [(face2[1], face2[0] + face2[2], face2[1] + face2[3], face2[0])]
        encodings2 = face_recognition.face_encodings(image2, face_locations2)
        if len(encodings2) == 0:
            print("No encoding generated for face2.")
        else:
            encoding2 = encodings2[0]
    
    return encoding1, encoding2

def compute_distances(encoding1, encoding2):
    if encoding1 is None or encoding2 is None:
        print("One or both encodings are invalid. Cannot compute distance.")
        return None
    
    return np.linalg.norm(np.array(encoding1) - np.array(encoding2))

def match_face_encodings(distance, threshold=0.6):
    if distance is None:
        print("Invalid distance value. Cannot determine a match.")
        return False

    if distance <= threshold:
        print(f"Faces match! Distance: {distance} (Threshold: {threshold})")
        return True
    else:
        print(f"Faces do not match. Distance: {distance} (Threshold: {threshold})")
        return False

###Main execution
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Face authentication comparison tool')
    parser.add_argument('image1', type=str, help='Path to first image')
    parser.add_argument('image2', type=str, help='Path to second image')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate paths exist
    if not os.path.exists(args.image1):
        print(f"Error: Image file '{args.image1}' does not exist")
        exit(1)
    if not os.path.exists(args.image2):
        print(f"Error: Image file '{args.image2}' does not exist")
        exit(1)
        
    # Use the provided image paths
    image1 = args.image1
    image2 = args.image2

    # Step 1: Detect faces in both images
    face1, face2 = detect_face_haar(image1, image2)
    if face1 is None or face2 is None:
        print("Face detection failed for one or both images. Exiting.")
        exit()

    # Step 2: Generate face encodings for both images
    encoding1, encoding2 = get_face_encodings(image1, face1, image2, face2)
    if encoding1 is None or encoding2 is None:
        print("Face encoding failed for one or both faces. Exiting.")
        exit()

    # Step 3: Compute the Euclidean distance between the two encodings
    distance = compute_distances(encoding1, encoding2)
    if distance is None:
        print("Distance computation failed. Exiting.")
        exit()

    # Step 4: Compare the distance against the threshold to determine a match
    is_match = match_face_encodings(distance, threshold=0.6)
    if is_match:
        print("The faces match!")
    else:
        print("The faces do not match.")
