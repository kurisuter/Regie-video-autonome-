import cv2,time
import sys
import cv2.cv as cv
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

init()
cascPath ="haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
print faceCascade.empty()
cascade = faceCascade
nested = cv2.CascadeClassifier("haarcascade_eye")
print nested.empty()
video_capture = cv2.VideoCapture(1)
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

    

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    rects = detect(gray, cascade)
    vis = frame.copy()
    recad = vis
    # Draw a rectangle around the faces
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 5)
    for x1, y1, x2, y2 in rects:
            center(frame,x1,x2,y1,y2)
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = detect(roi.copy(), nested)
            #draw_rects(vis_roi, subrects, (255, 0, 0))
            recad = roi
            for x1, y1, x2, y2 in subrects:
                cv2.rectangle(vis_roi, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    
    # Display the resulting frame
    cv2.imshow('Video', vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
