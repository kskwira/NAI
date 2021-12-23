import cv2 as cv

cascPath = 'haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(cascPath)

if face_cascade.empty():
    raise IOError('Unable to load the cascade classifier xml file')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rect = face_cascade.detectMultiScale(grey_frame, 1.3, 5)

    for (x, y, w, h) in face_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv.imshow("Face detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
