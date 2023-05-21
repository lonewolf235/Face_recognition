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
l=[]
for detection in detections:
         # Extract face ROI
         x, y, w, h = detection['box']
         face = img[y:y+h, x:x+w]
         print(type(face))
         l.append(face)
         cv2.imwrite(unknown_faces_dir+"image_new"+str(count)+".jpg", face)
         count+=1

for i in l:
    result=DeepFace.find(i,known_faces_dir,model_name='Facenet', enforce_detection=False, detector_backend='retinaface')
    print(result)