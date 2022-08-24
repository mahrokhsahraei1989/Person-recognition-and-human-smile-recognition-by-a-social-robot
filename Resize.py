#import the library opencv
import cv2
#globbing utility.
import glob
#select the path
#I have provided my path from my local computer, please change it accordingly
path = "C:/Users/akbari/Desktop/python_peactice/dataset1/sahraei/*.*"
path2 = "C:/Users/akbari/Desktop/python_peactice/dataset/Sahraei/"
N=1
for file in glob.glob(path):
    im = cv2.imread(file)
    resized = cv2.resize(im, (400,400), interpolation = cv2.INTER_CUBIC)
    name = path2+"im"+str(N)+".jpg"
    cv2.imwrite(name,resized)
    N=N+1
    
    