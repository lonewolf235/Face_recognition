from deepface import DeepFace
from mtcnn import MTCNN
import cv2
import os
from datetime import datetime
import matplotlib.pyplot as plt


known_faces_dir='/home/nihal/Prakash/face_recognition/known_faces'
unknown_faces_dir = 'unknown_faces/'

detector= MTCNN(min_face_size=20, scale_factor=0.709)

known_faces = {}
for filename in os.listdir(known_faces_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() in ['.jpg', '.jpeg', '.png']:
        filepath = os.path.join(known_faces_dir, filename)
        face_image = cv2.imread(filepath)
        known_faces[name] = face_image

img_path='/home/nihal/Prakash/face_recognition/Linkin-Park.jpg'
img=plt.imread(img_path)
detections = detector.detect_faces(img)
print(len(detections))
print(detections)

    # Loop through each face and recognize the person
count=0
for detection in detections:
         # Extract face ROI
         x, y, w, h = detection['box']
         face = img[y:y+h, x:x+w]
         print(type(face))
         cv2.imwrite(unknown_faces_dir+"image_new"+str(count)+".jpg", face)
         count+=1



# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     # Detect faces in the frame
#     detections = detector.detect_faces(frame)
#     print(type(detections))
#     print(detections)

#     # Loop through each face and recognize the person
#     for detection in detections:
#          # Extract face ROI
#          x, y, w, h = detection['box']
#          face = frame[y:y+h, x:x+w]
#          cv2.imwrite("image_name.jpg", face)
#     #     # Recognize the person in the face
#     #     result = DeepFace.verify(face, known_faces.values(), model_name='Facenet', enforce_detection=False)

#     #     # Draw bounding box around the face
#     #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     #     # Display name of the recognized person
#     #     if result['verified']:
#     #         cv2.putText(frame, result['verified'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     #     else:
#     #         # Save the unknown face as a new image file
#     #         timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#     #         filename = f'unknown_face_{timestamp}.jpg'
#     #         filepath = os.path.join(unknown_faces_dir, filename)
#     #         cv2.imwrite(filepath, face)
#         # print("hello")
#         # output.write(frame)
#     # Display the resulting frame
#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the window
# video_capture.release()
# cv2.destroyAllWindows()