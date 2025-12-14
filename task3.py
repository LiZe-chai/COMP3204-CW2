import cv2

camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = camera.get(cv2.CAP_PROP_FPS)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter("video_captured.mp4", fourcc, fps, (width, height), isColor=False)

win_name = "Webcam - Grayscale Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


while (cv2.waitKey(1) & 0xFF) != ord('q'):   # Escape when 'q' is pressed
    has_frame, frame = camera.read()
    if not has_frame:
        break
    # convert to grayscale
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    output.write(frame)
    cv2.imshow(win_name, frame)

camera.release()
output.release()
cv2.destroyWindow(win_name)