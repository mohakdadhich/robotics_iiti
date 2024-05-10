import cv2 as cv

haarcas= "Robotics\haarcascade_frontalface_default.xml"
def detectFace(img_path):
    img = cv.imread(img_path)
    face_cascade = cv.CascadeClassifier(haarcas)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(4,4))      # for small face pics
    # face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=7, minSize=(15,15))      # for lagre face pics
    for (x,y,w,h) in face:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv.imshow("detected face", img)
    print(f"Total faces = {len(face)}")
    cv.waitKey(0) & 0xFF == ord('q')
    cv.destroyAllWindows()

img_path = "Robotics\\face1.png"
detectFace(img_path)