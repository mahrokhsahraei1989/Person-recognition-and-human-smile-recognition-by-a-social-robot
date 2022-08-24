# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import imutils
import argparse
import pickle
import cv2
import numpy as np
import time
from imutils.video import VideoStream
import datetime
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
SVM = svm.load("SVM_XML/Trained_svm_10_10_10_30.xml")

vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)
# loop over frames from the video file stream
while True:
#    T1 = datetime.datetime.now();
    	# grab the frame from the threaded video stream
    image = vs.read()
    image = imutils.resize(image, width=300)
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
            
    
    
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        if name == "Sahraei":
            
    	# draw the predicted face name on the image
    	    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
    	    y = top - 15 if top - 15 > 15 else top + 15
    	    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
    		    0.25, (0, 255, 0), 1)
        else:
            	# draw the predicted face name on the image
    	    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
    	    y = top - 15 if top - 15 > 15 else top + 15
    	    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
    		    0.25, (0, 0, 255), 1)
#    T5 = datetime.datetime.now();	
#    T45 = T5 - T4;
#    print("T45 is " + str ( int(T45.total_seconds()*1000)))
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(1)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()