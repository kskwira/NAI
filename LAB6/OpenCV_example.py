"""
Authors: Krzysztof Skwira & Tomasz Lemke
See README.md for description
"""
import cv2 as cv

cascPath = 'haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(cascPath)
background = None

# checking if file with Haar Cascade was uploaded correctly
if face_cascade.empty():
    raise IOError('Unable to load the cascade classifier xml file')

# Video capture
cap = cv.VideoCapture(0)

# While loop for endless video capture
while True:
    _, frame = cap.read()
    _, frame2 = cap.read()

    diff = cv.absdiff(frame, frame2)  # calculating differences between 2 vid captures
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)  # change to grayscale
    blur = cv.GaussianBlur(gray, (5, 5), 0)  # calculating the Gaussian blur
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)  # assignment of pixel values in relation to the threshold value provided
    dilated = cv.dilate(thresh, None, iterations=3)  # dilating image
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # capturing contours of objects
    face_aim = face_cascade.detectMultiScale(frame, 1.3, 5)  # implementing face detection from Haar Cascade
    for contour in contours:
        for (x, y, w, h) in face_aim:

            if cv.contourArea(contour) < 400:  # setting the movement sensitivity
                continue
            color = (0, 0, 255)
            thickness = 2
            isClosed = False
            center_coordinates = x + w // 2, y + h // 2  # finding center of the face
            radius = w + 5 // 2  # or can be h / 2 or can be anything based on your requirements

            cv.circle(frame, center_coordinates, radius, (0, 0, 255), 3)  # drawing circle around face
            cv.line(frame, (x + w - 400 // 2, y + h // 2), (x + w + 200 // 2, y + h // 2), color,
                    thickness)  # horizontal line
            cv.line(frame, (x + w // 2, y + h - 400 // 2), (x + w // 2, y + h + 200 // 2), color,
                    thickness)  # vertical line

    cv.imshow("Face detection", frame)  # video display
    if cv.waitKey(1) & 0xFF == ord('q'):  # exit condition
        break

cap.release()
cv.destroyAllWindows()
