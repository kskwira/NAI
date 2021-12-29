import cv2 as cv

cascPath = 'haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(cascPath)
background = None


if face_cascade.empty():
    raise IOError('Unable to load the cascade classifier xml file')

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    _, frame2 = cap.read()

    diff = cv.absdiff(frame, frame2)
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=3)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    face_aim = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in face_aim:
        for contour in contours:
            # (x, y, w, h) = cv.boundingRect(contour)

            if cv.contourArea(contour) < 400:
                continue
            color = (0, 0, 255)
            thickness = 2
            isClosed = False
            center_coordinates = x + w // 2, y + h // 2
            radius = w + 5 // 2  # or can be h / 2 or can be anything based on your requirements

            cv.circle(frame, center_coordinates, radius, (0, 0, 255), 3)

            cv.line(frame, (x + w - 400 // 2, y + h // 2), (x + w + 200 // 2, y + h // 2), color, thickness)  # horizontal line
            cv.line(frame, (x + w // 2, y + h - 400 // 2), (x + w // 2, y + h + 200 // 2), color, thickness)  # vertical line

    cv.imshow("Face detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
