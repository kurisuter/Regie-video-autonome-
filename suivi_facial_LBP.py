import sys, cv2, os
from cv2 import cv
import numpy as np
import cv2,time
import serial

ser = serial.Serial('COM5', 9600, timeout=1)  # open first serial port
print ser.name          # check which port was really used

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def move(servo, angle):
    '''Moves the specified servo to the supplied angle.

    Arguments:
        servo
          the servo number to command, an integer from 1-4
        angle
          the desired servo angle, an integer from 0 to 180

    (e.g.) >>> servo.move(2, 90)
           ... # "move servo #2 to 90 degrees"'''

    if (0 <= angle <= 180):
        ser.write(chr(255))
        ser.write(chr(servo))
        ser.write(chr(angle))
    else:
        print("Servo angle must be an integer between 0 and 180.\n")


def init():
    move(9,90)
    move(6,90)
    
def read_csv(filename,images,labels):
    print filename
    try:
        file = open(filename,"r")
    except OSError:
        sys.stderr.write("Failed to open dirctory %s\n" 
                        % filename)
        return -1
    
    tab = file.read().split("\n")
    for ligne in tab:
        t = ligne.split(";")
        if len(t)>= 2:
            path = t[0]
            label = t[1]
            print path
            print label
            images.append(cv2.imread(path,0))
            labels.append(int(label))
    file.close()
    
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects
init()

angx = 90
angy = 90
def center(frame,x1,x2,y1,y2):
    global angx
    global angy
    print "largeur = "+str(frame.shape[0]/2)
    print "hauteur = "+str(frame.shape[1]/2)
    print "visage_X = "+str(((x2-x1)/2)+x1)
    print "visage_Y = "+str(((y2-y1)/2)+y1)
    
    if (frame.shape[1]/2)> (((x2-x1)/2)+x1+10) and angx <180:
        angx= angx + 1
        move(6,angx)
    elif(frame.shape[1]/2)< (((x2-x1)/2)+x1-10) and angx >0:
        angx= angx - 1
        move(6,angx)
    if (frame.shape[0]/2)> (((y2-y1)/2)+y1+10) and angy <180:
        angy= angy + 1
        move(9,angy)
    elif(frame.shape[0]/2)< (((y2-y1)/2)+y1-10) and angy >0:
        angy= angy - 1
        move(9,angy)
def main():
    # Check for valid command line arguments, print usage
    # if no arguments were given.
   # if (len(sys.argv) != 4):
    #    sys.stderr.write("usage %s </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>\n\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection.\n\t </path/to/csv.ext> -- Path to the CSV file with the face database. \n\t <device id> -- The webcam device id to grab frames from."  % sys.argv[0])
     #   return -1
    
    # Get the path to your CSV:
    fn_haar2 = "C:\Users\Christelle\Downloads\opencv\sources\data\haarcascades\haarcascade_profileface.xml"
    fn_haar ="haarcascade_frontalface_alt.xml" #"C:\Users\Christelle\Downloads\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml" #str(sys.argv[1])      # string
    fn_csv =  "D:\documents\Polytech_2014-2015\Semestre_8\Projet\\at.csv" #str(sysargv[2])        # string str(sys.argv[1])#
    deviceId = 1# int(sys.argv[3])     # int

    # These vectors hold the images and corresponding labels:
    images=[]  #vector<Mat>
    labels=[]   #vector<int>

    # Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try:
        read_csv(fn_csv,images,labels)
    except OSError:
        sys.stderr.write("Failed to open dirctory %s\n" 
                        % filename)
        return -1

    # Get the height from the first image. We'll need this
    # later in code to reshape the images to their original
    # size AND we need to reshape incoming faces to this size:
    im_width = images[0].shape[1]
    im_height = images[0].shape[0]
    labels = np.array(labels)
    # Create a FaceRecognizer and train it on the given images:
    model = cv2.createLBPHFaceRecognizer()    # Ptr<FaceRecognizer> 
    model.train(images, labels)
    
    # That's it for learning the Face Recognition model. You now
    # need to create the classifier for the task of Face Detection.
    # We are going to use the haar cascade you have specified in the
    # command line arguments:
    haar_cascade= cv2.CascadeClassifier(fn_haar)
    haar_cascade2= cv2.CascadeClassifier(fn_haar2)
    print haar_cascade.empty()
    # Get a handle to the Video device:
    cap = cv2.VideoCapture(deviceId)
    
    # Check if we can use this device at all:
    if (not cap.isOpened()):
        print 'Warning: unable to open video source: ', deviceId
        
    # Holds the current frame from the Video device:
    while True:
        ret, frame = cap.read()
        # Clone the current frame:
        original = frame.copy()
        # Convert the current frame to grayscale:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY);
        gray = cv2.equalizeHist(gray)
        # Find the faces in the frame:
        faces = detect(gray,haar_cascade)
        profiles = detect(gray,haar_cascade2)
        for x1, y1, x2, y2 in faces:
            center(frame,x1,x2,y1,y2)
            face = gray[y1:y2, x1:x2]
            face_resized=cv2.resize(face, (im_height,im_width),fx = 1.0, fy = 1.0, interpolation = cv2.INTER_CUBIC)

            # Now perform the prediction, see how easy that is:
            prediction = model.predict(face_resized)
            
            # And finally write all we've found out to the original image!
            # First of all draw a green rectangle around the detected face:
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Create the text we will annotate the box with:
            if (prediction[0]==1):
                box_text = "Prediction = Jean"+str(prediction[1])
            elif (prediction[0]==2):
                box_text = "Prediction = Laurent  "+str(prediction[1])
            elif (prediction[0]==4):
                box_text = "Prediction = Christelle  "+str(prediction[1])
            elif (prediction[0]==5):
                box_text = "Prediction = Tara  "+str(prediction[1])
            else:
                box_text = "Prediction = Autre  "+str(prediction)
            
            # Calculate the position for annotated text (make sure we don't
            # put illegal values in there):
            pos_x = int( max(x1 - 10, 0))
            pos_y = int( max(y1 - 10, 0))
            # And now put it into the image:
            cv2.putText(original, box_text, (pos_x,pos_y),cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0));
        for x1, y1, x2, y2 in profiles:
            center(frame,x1,x2,y1,y2)
            prof = gray[y1:y2, x1:x2]
            prof_resized=cv2.resize(prof, (im_height,im_width),fx = 1.0, fy = 1.0, interpolation = cv2.INTER_CUBIC)

            # Now perform the prediction, see how easy that is:
            prediction = model.predict(prof_resized)
            
            # And finally write all we've found out to the original image!
            # First of all draw a green rectangle around the detected face:
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Create the text we will annotate the box with:
            if (prediction[0]==1):
                box_text = "Prediction = Jean"+str(prediction[1])
            elif (prediction[0]==2):
                box_text = "Prediction = Laurent  "+str(prediction[1])
            elif (prediction[0]==4):
                box_text = "Prediction = Christelle  "+str(prediction[1])
            elif (prediction[0]==5):
                box_text = "Prediction = Tara  "+str(prediction[1])
            else:
                box_text = "Prediction = Autre  "+str(prediction)
            
            # Calculate the position for annotated text (make sure we don't
            # put illegal values in there):
            pos_x = int( max(x1 - 10, 0))
            pos_y = int( max(y1 - 10, 0))
            # And now put it into the image:
            cv2.putText(original, box_text, (pos_x,pos_y),cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0));
            
        # Display the resulting frame
        cv2.imshow('Video', original)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
        main()
