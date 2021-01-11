import numpy as numpy
import cv2
import pickle
import os
import PIL
from PIL import Image
import numpy as np
import time
import shutil
import datetime

PERSONE_ID={}
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR, "images")

# --- You can select various .xml file with different recognition features: 
    #face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    #face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
    #face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt_tree.xml')
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer.read("trainer.yml")
current_id=0
label_ids={}
y_labels=[]
x_train=[]

print("\n\n____ ___  ____ _  _ ____ _  _    ___  _   _ _  _ ___ _  _ ____ _  _\n|  | |__] |___ |\ | |    |  |    |__]  \_/  |__|  |  |__| |  | |\ |\n|__| |    |___ | \| |___  \/     |      |   |  |  |  |  | |__| | \| \n\n\n\n\n\n\n____ ____ ____ ____    ____ ____ ____ ____ ____ _  _ _ ___ _ ____ _  _ \n|___ |__| |    |___    |__/ |___ |    |  | | __ |\ | |  |  | |  | |\ | \n|    |  | |___ |___    |  \ |___ |___ |__| |__] | \| |  |  | |__| | \| \n\n\n") #Start Title
richiesta=input("Update recognition files? [Y/N]:  ")
if richiesta=='Y' or richiesta=='y':
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"): 
                path=os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id+=1
                    
                id_ = label_ids[label]
                pil_image=Image.open(path).convert("L")
                size=(550,550)
                final_image=pil_image.resize(size, Image.ANTIALIAS)
                image_array=np.array(pil_image, "uint8") #pil is the previous one
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
    with open("pickles/face-labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")

numero = 1
unlocking= False
labels={"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
cap=cv2.VideoCapture(0)
print(label_ids)
num=1
oldn=str
while(unlocking == False):
    ret, frame =cap.read() #capture frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)  # scale factor more high more is accurate
    for (x, y, w, h) in faces:
        roi_frame =gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_frame)
        if conf>=4 and conf <=85:
            print("Ciao ", labels[id_], " ", num)
            if not labels[id_] in PERSONE_ID:
                PERSONE_ID[labels[id_]]= 1
            else:
                PERSONE_ID[labels[id_]]+=1
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (250, 250, 250)
            num+=1
            #After 30 Recognitions it creates a copy of the frame captured
            '''if num%30==0:
                Current_Date = str(datetime.datetime.today().strftime ('%d-%b-%Y%H-%M-%S'))
                os.rename(r'MyImage.png', r'YOUR DIRECTORY \images\\' + labels[id_] + '\\' + str(Current_Date) + '.png') ''' 
            if num%7==0:
                spessore=2
                cv2.putText(frame, name, (x,y), font, 1, color, spessore, cv2.LINE_AA)
        else:
            num=1
        img_item= "MyImage.png"
        cv2.imwrite(img_item, roi_frame)
        oldn=label_ids
        color = (255,0,0) #BGR color
        spessore = 3
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x,y), (width, height), color, spessore)
        
    cv2.imshow('frame', frame) # To Show the real-time webcam frame
    if num%50==0:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n----------------------------------\nSuccessful User Authentication: " + labels[id_] + "\n\n----------------------------------")
        break

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#To print recognized faces
'''print("\n----------------------------------\nRecognized Faces: \n")
print(PERSONE_ID)
print("\n----------------------------------\n")'''
richiesta=input("\nPress enter to quit")
os.system('cls' if os.name == 'nt' else 'clear')