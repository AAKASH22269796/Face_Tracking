import cv2
import mediapipe as mp

mp_face_detection=mp.solutions.face_detection
face_detection=mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5)
mp_drawing=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

while cap.isOpened():
    success,image=cap.read()
    if not success:
        print("Ignore empty frame.")
        continue
    image=cv2.flip(image,1)
    rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result=face_detection.process(rgb_image)

    if result.detections:
        for detection in result.detections:
            mp_drawing.draw_detection(image,detection)
    cv2.imshow("Face Tracking",image)

    if cv2.waitKey(1) & 0xFF==ord('c'):
        break

cap.release()
cv2.destroyAllWindows()

