# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import face_recognition
import imutils
import argparse
import pickle
import cv2
import numpy as np
import time
from imutils.video import VideoStream

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", type=str, default="encodings.pickle",
	help="path to serialized db of facial encodings")
#ap.add_argument("-i", "--image", type=str, default="index3.jpg",
#	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())


svm = cv2.ml.SVM_create()
SVM = svm.load("SVM_XML/Trained_svm_1_10_10_30.xml")


model_smile = load_model("lenet15.hdf5")


vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)
# loop over frames from the video file stream
while True:
#    T1 = datetime.datetime.now();
    	# grab the frame from the threaded video stream
    image = vs.read()
    image = imutils.resize(image, width=200)
#    T2 = datetime.datetime.now();	
#    T12 = T2 - T1;
#    print("T12 is " + str ( int(T12.total_seconds()*1000)))
    	# convert the input frame from BGR to RGB then resize it to have
    	# a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
    
#    T3 = datetime.datetime.now();	
#    T23 = T3 - T2;
#    print("T23 is " + str ( int(T23.total_seconds()*1000)))
    
    boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes,1)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Labels_S=[]
    for (top1, right1, bottom1, left1) in boxes:
        roi = gray[top1:bottom1, left1:right1]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)
        (notSmiling, smiling) = model_smile.predict(roi)[0]
        label_smile = "Smiling" if smiling >= notSmiling else "Not_Smiling"
        Labels_S.append(label_smile)
    
#    T4 = datetime.datetime.now();	
#    T34 = T4 - T3;
#    print("T34 is " + str ( int(T34.total_seconds()*1000)))
    #Result = SVM.predict(ff)
    
    names = []
    for encoding in encodings :
        Result = SVM.predict(np.matrix(encoding, dtype=np.float32))[1]
        print("Result is " + str(Result) )
        if Result == 1.0:
            names.append("Sahraei")
        else :
            names.append("Other")
            
    
    
    N=0 
    # loop over the recognized faces
    Text1  = "Sahraei_Smiling"
    Text2  = "Sahraei_Not_Smiling"
    Text3  = "Other_Smiling"
    Text4  = "Other_Not_Smiling"
    for ((top, right, bottom, left), name) in zip(boxes, names):
        
        if ( name == "Sahraei" and Labels_S[N] == "Smiling" ) :
            
    	    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
    	    y = top - 15 if top - 15 > 15 else top + 15
    	    cv2.putText(image, Text1, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
    		    0.25, (0, 0, 255), 1)
            
        elif ( name == "Sahraei" and Labels_S[N] == "Not_Smiling" ):   
        
    	    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
    	    y = top - 15 if top - 15 > 15 else top + 15
    	    cv2.putText(image, Text2, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
    		    0.25, (0, 255, 0), 1)
            
        elif ( name == "Other" and Labels_S[N] == "Smiling" ):   
         	
        	    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
        	    y = top - 15 if top - 15 > 15 else top + 15
        	    cv2.putText(image, Text3, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        		    0.25, (255, 0, 255), 1)
     
        elif ( name == "Other" and Labels_S[N] == "Not_Smiling" ):   
            
        	    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
        	    y = top - 15 if top - 15 > 15 else top + 15
        	    cv2.putText(image, Text4, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        		    0.25, (255, 0, 0), 1)   
        N = N + 1        
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(1)
   

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()