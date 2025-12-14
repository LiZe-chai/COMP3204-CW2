import cv2

camera = cv2.VideoCapture(0)
win_name = "Face detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

while (cv2.waitKey(1) & 0xFF) != ord('q'):      # Escape when 'q' is pressed
    has_frame, frame = camera.read()
    if not has_frame:
        break
    # convert to grayscale
    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=10)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow(win_name, frame)


camera.release()
cv2.destroyWindow(win_name)