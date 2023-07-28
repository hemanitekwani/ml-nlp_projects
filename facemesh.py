import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

while True:
 
    ret, frame = cap.read()

    if not ret:
        break


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
               
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

               
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

   
    cv2.imshow('Face Mesh', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()