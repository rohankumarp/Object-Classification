import cvzone
import cv2

cap = cv2.VideoCapture(1)
myclassifier = cvzone.Classifier('modeldata(updated)/keras_model.h5' , 'modeldata(updated)/labels.txt')

while True :
    ret,img = cap.read()
    myclassifier.getPrediction(img,scale=1 , pos=(-20,70))

    cv2.imshow('Image' , img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
cv2.destroyAllWindows()