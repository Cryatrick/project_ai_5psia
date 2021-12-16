import cv2
import os
import numpy as np
from PIL import Image

cam = cv2.VideoCapture(0)


cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

saved_names = open("name_data.txt","a")

file1 = open('name_data.txt', 'r')
Lines = file1.readlines()
face_id = len(Lines)
# print(last_line);
 # For each person, enter one numeric face id
# face_id = input('\n enter user id and press Enter ==>  ')
name = input('\n Enter Name Please ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
saved_names.write("\n"+str(face_id)+";"+str(name));
# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        # Save the captured image into the datasets folder
        cv2.imwrite("datasets/User." + str(face_id) + '.' +  str(count) + ".jpg", gray[y:y+h,x:x+w])
        count += 1
    cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
# Path for face image database
path = 'datasets'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml");
# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

cv2.destroyAllWindows()