import cv2

video_path = r"C:\Users\chail\OneDrive - University of Southampton\COMP 3204\CW2\CCTV.mp4"
cap = cv2.VideoCapture(video_path)

backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

if not cap.isOpened():
    print("Error opening video file")
    exit()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter("contour_detection.mp4", fourcc, fps, (width, height))

motion = False
start_time = 0
start_threshold = 10
end_threshold = 5
appear_count = 0
disappear_count = 0


while cap.isOpened():

    has_frame, frame = cap.read()
    if not has_frame:
        break

    #preprocess frame
    blur = cv2.GaussianBlur(frame, (3, 3), 4)
    # background subtraction
    fg_mask = backSub.apply(blur)

    # threshold to get binary mask
    retval, mask_thresh = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)

    # morphology operation to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter small motion
    min_contour_area = 13000
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    if len(large_contours) > 0:
        appear_count += 1
        disappear_count = 0
    else:
        disappear_count += 1
        appear_count = 0
    
    # New motion detected
    if not motion and appear_count >= start_threshold:
        motion = True
        start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        print(f"Motion start at {start_time:.2f} sec")

    
    if motion and disappear_count >= end_threshold:
        motion = False
        end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        duration = end_time - start_time
        print(f"Motion end at {end_time:.2f} sec")
        print(f"Duration {duration:.2f} sec\n")

    # draw bounding boxes
    frame_out = frame.copy()
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame_out, (x, y), (x+w, y+h), (0, 0, 255), 3)
    
    output.write(frame_out)


    cv2.imshow("Output", frame_out)

    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break


cap.release()
output.release()
cv2.destroyAllWindows()
