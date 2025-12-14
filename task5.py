import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 10
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter("face_emotion_recognition.mp4", fourcc, fps, (width, height))

currentEmotion =  "No face detected"
startTime = datetime.now()

df = pd.DataFrame(columns=["emotion", "start_time", "end_time", "duration"])

pendingEmotion = None
pendingStart = None
hysteresis_seconds = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_frame, (3, 3), 1)

    faces = face_cascade.detectMultiScale(blur, 1.1, 5)

    emotion = "No face detected"

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # DeepFace emotion detection
        result = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False)
        emotion = result[0]["dominant_emotion"]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame,"face",(x, y -10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 50)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
            cv2.putText(frame,"eyes",(x + ex,y + ey -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        smiles = smile_cascade.detectMultiScale(face_gray, 1.1, 50)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)
            cv2.putText(frame,"smile",(x + sx,y + sy -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    if emotion != currentEmotion:
        if pendingEmotion != emotion:
            pendingEmotion = emotion
            pendingStart = datetime.now()
        else:
            elapsed = (datetime.now() - pendingStart).total_seconds()
            if elapsed >= hysteresis_seconds:
                if currentEmotion is not None:
                    endTime = datetime.now()
                    duration = (endTime - startTime).total_seconds()
                    df.loc[len(df)] = [
                        currentEmotion,
                        startTime.strftime("%Y-%m-%d %H:%M:%S"),
                        endTime.strftime("%Y-%m-%d %H:%M:%S"),
                        duration
                    ]

                currentEmotion = emotion
                startTime = datetime.now()
                pendingEmotion = None
                pendingStart = None
    else:
        pendingEmotion = None
        pendingStart = None

    cv2.putText(frame, f"current emotion: {currentEmotion}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    output.write(frame)

    cv2.imshow("Face Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close last emotion
endTime = datetime.now()
df.loc[len(df)] = [currentEmotion, startTime.strftime("%Y-%m-%d %H:%M:%S"), endTime.strftime("%Y-%m-%d %H:%M:%S"), duration]

cap.release()
output.release()
cv2.destroyAllWindows()

df.to_csv("emotion_tracking.csv", index=False)

emotion_duration = df.groupby("emotion")["duration"].sum()
emotion_duration.plot(kind="bar", figsize=(8,5))
plt.xlabel("Emotion")
plt.ylabel("Duration (seconds)")
plt.tight_layout()
plt.show()