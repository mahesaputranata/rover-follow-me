import cv2
import face_recognition
import numpy as np
import os

SAVE_PATH = "data/owner_face.npy"
SAMPLE_COUNT = 30

def enroll():
    cap = cv2.VideoCapture(0)
    embeddings = []

    print("Look at the camera")

    while len(embeddings) < SAMPLE_COUNT:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # rgb = frame[:,:,::-1]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        locations = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, locations)

        if encs:
            embeddings.append(encs[0])
            print(f"Sample {len(embeddings)}/{SAMPLE_COUNT}")
        
        cv2.imshow("Enrollment, look at the camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    if embeddings:
        avg = np.mean(embeddings, axis = 0)
        os.makedirs("data", exist_ok=True)
        np.save(SAVE_PATH, avg)
        print(f"Saved to {SAVE_PATH}")
    else:
        print("No face detected, please try again")
    
if __name__ == "__main__":
    enroll()
        