import cv2
import numpy as np
from sklearn.cluster import KMeans

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_dominant_color(image, k=4):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    return kmeans.cluster_centers_[kmeans.labels_[0]]

def color_distance(c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2))

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def draw_labels(frame, faces, color):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def main():
    logo_image_path = r'C:\Users\Hey!\OneDrive\Desktop\Tuesday\uniform_logo.jpg'
    logo_color = extract_dominant_color(cv2.imread(logo_image_path))

    cap = cv2.VideoCapture(0)  # Change to 0 for default camera or adjust as needed

    uniform_detected = False  # To track if uniform has been detected

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Extract dominant color from the frame
        uniforms_color = extract_dominant_color(frame)

        # Check color match and mark presence if needed
        if not uniform_detected and color_distance(logo_color, uniforms_color) < 50:
            cv2.putText(frame, 'Uniform Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            uniform_detected = True
            print("Attendance Marked!")

            # Save the snapshot and exit the loop
            snapshot_path = 'attendance_snapshot.jpg'
            cv2.imwrite(snapshot_path, frame)
            print(f"Snapshot saved to {snapshot_path}")
            break
        else:
            cv2.putText(frame, 'No Uniform Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            uniform_detected = False  # Reset detection if no uniform is detected

        # Detect faces and highlight based on uniform detection status
        faces = detect_faces(frame)
        face_color = (0, 255, 0) if uniform_detected else (0, 0, 255)
        draw_labels(frame, faces, face_color)

        # Display the frame
        cv2.imshow('Detection', frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
