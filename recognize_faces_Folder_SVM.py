# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import numpy as np
import os


# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())

svm = cv2.ml.SVM_create()
SVM = svm.load("SVM_XML/Trained_svm_0.1_10_10_30.xml")

ImagePaths = list(paths.list_images("Test"))
Correct = 0
Incorrect = 0

for (n,Image_Path) in enumerate(ImagePaths):
    print("ProcessingImae {}/{}.".format(n+1,len(ImagePaths)))
    Name = Image_Path.split(os.path.sep)[-2]
    Image = cv2.imread(Image_Path)
    RGB = cv2.cvtColor(Image,cv2.COLOR_BGR2RGB)
    Boxes = face_recognition.face_locations(RGB,model="hog")
    if len(Boxes) == 0 or len(Boxes) > 1:
        continue
    Encoding = face_recognition.face_encodings(RGB,Boxes,1)
    Result = SVM.predict(np.matrix(Encoding, dtype=np.float32))[1]
    print(Name + " is " + str(Result)) 
    if Result == 1 :
        Predict_Name = "Sahraei"
    else :
        Predict_Name = "Other"
        
    if Name == Predict_Name:
        Correct = Correct + 1
    else :
        Incorrect = Incorrect+ 1
        
     

print(Correct)
print(Incorrect)
print(Correct / ( Correct + Incorrect))





