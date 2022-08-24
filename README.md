# Person-recognition-and-human-smile-recognition-by-a-social-robot
Person recognition and human smile recognition by a social robot with algorithms: &amp;HOG, MMOD, Fast Cascade Viola &amp; Jones, Residual Neural Networks

Person recognition and human smile recognition by a social robot with algorithms: &HOG, MMOD, Fast Cascade Viola & Jones, Residual Neural Networks
This code is written to make the behavior of a social robot intelligent. This interactive robot can communicate with humans through vision.
Work steps:
First, create two folders of your images (target images and non-target images). There are target images of your own images in different dimensions and lights, which should be more than 500 in order to increase the accuracy of the algorithm. And the images in the folder of others include people whose faces are not similar to yours.
resize: to resize images
encode_faces.py: to extract features from training images
encodings.pickle: file extracted from features
recognize_faces_Folder_SVM: to recognize and identify the target person among all the images in a folder
recognize_faces_image.py: to recognize and identify the target person from a photo
recognize_faces_video_SVM.py: Recognize the target person in a video file from all the people in the video
recognize_faces_video_SVM (3)_smile.py: Recognize the target person's smile from all the people standing in front of the camera.
