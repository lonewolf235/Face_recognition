from deepface import DeepFace
from mtcnn import MTCNN
import cv2
import os
from datetime import datetime



# Load MTCNN face detection model
detector= MTCNN(min_face_size=20, scale_factor=0.709)

# Load the face recognition model

model = DeepFace.build_model('Facenet')

# Capture video from the default camera
video_capture = cv2.VideoCapture("/home/nihal/Prakash/face_recognition/Dubai Presents_ Shah Rukh Khan.mp4")

output = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MP4V'),25,(720,640))

# Specify the directory containing the known faces
known_faces_dir = 'known_faces/'

# Specify the directory to save the unknown faces
unknown_faces_dir = 'unknown_faces/'

# Create a new subdirectory for the unknown faces
os.makedirs(unknown_faces_dir, exist_ok=True)

# Load the known faces from the directory
known_faces = {}
for filename in os.listdir(known_faces_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() in ['.jpg', '.jpeg', '.png']:
        filepath = os.path.join(known_faces_dir, filename)
        face_image = cv2.imread(filepath)
        known_faces[name] = face_image

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Detect faces in the frame
    detections = detector.detect_faces(frame)

    # Loop through each face and recognize the person
    for detection in detections:
        # Extract face ROI
        x, y, w, h = detection['box']
        face = frame[y:y+h, x:x+w]

        # Recognize the person in the face
        result = DeepFace.verify(face, known_faces.values(), model_name='Facenet', enforce_detection=False)

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display name of the recognized person
        if result['verified']:
            cv2.putText(frame, result['verified'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Save the unknown face as a new image file
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'unknown_face_{timestamp}.jpg'
            filepath = os.path.join(unknown_faces_dir, filename)
            cv2.imwrite(filepath, face)
        print("hello")
        output.write(frame)
    # Display the resulting frame
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
