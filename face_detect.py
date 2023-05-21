from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import time


img_db="/home/nihal/Prakash/face_recognition/known_faces"
img_path='/home/nihal/Prakash/face_recognition/01.jpg'
detectors=['opencv','ssd','mtcnn']
count=0
for detector in detectors:
    tic=time.time()
    img=DeepFace.detectFace(img_path,detector_backend=detector)
    toc=time.time()
    DeepFace.find(img_path,img_db)
    plt.imsave('img'+str(count)+'.jpg',img)
    count+=1
    print(detector,"  backend  ",toc-tic," seconds")
    
