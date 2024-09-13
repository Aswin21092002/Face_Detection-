import cv2
import numpy as np
import pyttsx3
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def detect_logo(frame, logo_image):
    """Detect the company logo in the frame using template matching."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_logo = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(gray_frame, gray_logo, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    # Threshold for logo detection (Adjust as needed)
    threshold = 0.8
    return max_val >= threshold

def check_uniform_color(frame, color_lower, color_upper):
    """Check if the uniform color is present in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_lower, color_upper)
    color_presence = np.sum(mask) > 1000  # Adjust threshold as needed
    return color_presence

def save_video(filename):
    """Capture video from the webcam and save it to a file."""
    cap = cv2.VideoCapture(1)  # Use the correct index for your camera
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        speak("Error: Unable to access the camera.")
        return

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    print("Recording video. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from camera.")
            speak("Error: Unable to capture frame from camera.")
            break

        out.write(frame)
        cv2.imshow('Video Recording', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_video(video_path, logo_image, color_lower, color_upper):
    """Process the video file to check faces and uniform status, and save 'present' images."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        speak("Error: Unable to open video file.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create directory to save present images
    if not os.path.exists('present_images'):
        os.makedirs('present_images')

    present_frame_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        frame_count += 1
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            logo_detected = detect_logo(face_roi, logo_image)
            color_detected = check_uniform_color(face_roi, color_lower, color_upper)

            if logo_detected or color_detected:
                cv2.putText(frame, 'Present', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle for Present
                speak("Uniform is correct.")
                
                # Save the frame with the present face
                present_frame_count += 1
                present_image_path = f'present_images/present_frame_{present_frame_count}.jpg'
                cv2.imwrite(present_image_path, frame)
            else:
                cv2.putText(frame, 'Absent', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle for Absent
                speak("Uniform is incorrect.")

        # Display the resulting frame
        cv2.imshow('Uniform Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to record and process video."""
    # Paths
    video_filename = 'captured_video.avi'
    logo_image_path = r'C:\Users\Hey!\OneDrive\Desktop\Tuesday\uniform_logo.jpg'
    
    # Define uniform color in HSV space
    color_lower = np.array([100, 150, 150])  # Lower bound for blue
    color_upper = np.array([140, 255, 255])  # Upper bound for blue

    # Load the logo image
    logo_image = cv2.imread(logo_image_path)
    if logo_image is None:
        print("Error: Unable to load logo image.")
        speak("Error: Unable to load logo image.")
        return

    # Step 1: Save video
    save_video(video_filename)
    
    # Step 2: Process the saved video
    process_video(video_filename, logo_image, color_lower, color_upper)

if __name__ == "__main__":
    main()
