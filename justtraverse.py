from deepface import DeepFace
from mtcnn import MTCNN
import cv2
import os
from datetime import datetime
import matplotlib.pyplot as plt

known_face_dir='/home/nihal/Prakash/face_recognition/known_faces'
rec_path='/home/nihal/Prakash/face_recognition/unknown_faces'
single='/home/nihal/Prakash/face_recognition/unknown_faces/image_new0.jpg'
files=os.listdir(rec_path)
x=[]
for i in files:
    x.append(os.path.join(rec_path,i))

result=DeepFace.find(single,known_face_dir,model_name='Facenet', enforce_detection=False, detector_backend='retinaface')
print(result)



# for j in x:
#     result=DeepFace.find(j,known_face_dir,model_name='Facenet', enforce_detection=False)
#     print(j,result)
