import cv2
import numpy as np
import pyttsx3
import datetime
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
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Threshold for logo detection (Adjust as needed)
    threshold = 0.8
    return max_val >= threshold

def check_uniform_color(frame, color_lower, color_upper):
    """Check if the uniform color is present in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_lower, color_upper)
    color_presence = np.sum(mask) > 1000  # Adjust threshold as needed
    return color_presence

def mark_attendance():
    """Mark attendance by creating a file with the current date and time."""
    now = datetime.datetime.now()
    attendance_file = 'attendance.txt'
    with open(attendance_file, 'a') as file:
        file.write(f'Attendance marked at {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
    print("Attendance marked.")
    speak("Attendance marked.")

def capture_video(video_path, duration=10, camera_index=1):
    """Capture a video for a specified duration."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        speak("Error: Unable to access the camera.")
        return None

    # Set video resolution (Adjust based on your camera capability)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set video codec and output file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 720))

    # Capture video
    start_time = cv2.getTickCount()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from camera.")
            speak("Error: Unable to capture frame from camera.")
            break

        out.write(frame)
        cv2.imshow('Recording Video', frame)

        # Check if the duration has been met
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time > duration:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    """Main function to detect uniform and logo, then capture video and mark attendance."""
    # Path to the reference logo image
    logo_image_path = r'C:\Users\Hey!\OneDrive\Desktop\Tuesday\uniform_logo.jpg'
    
    # Define uniform color in HSV space
    # Example: Blue color range for detection
    color_lower = np.array([100, 150, 150])  # Lower bound for blue
    color_upper = np.array([140, 255, 255])  # Upper bound for blue

    # Load the logo image
    logo_image = cv2.imread(logo_image_path)
    if logo_image is None:
        print("Error: Unable to load logo image.")
        speak("Error: Unable to load logo image.")
        return

    # Open the webcam
    cap = cv2.VideoCapture(0)  # Use the correct index for your camera
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        speak("Error: Unable to access the camera.")
        return

    video_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from camera.")
            speak("Error: Unable to capture frame from camera.")
            break

        # Detect the company logo
        logo_detected = detect_logo(frame, logo_image)
        
        # Check for uniform color
        color_detected = check_uniform_color(frame, color_lower, color_upper)
        
        if logo_detected or color_detected:
            cv2.putText(frame, 'Uniform Correct', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            speak("Uniform is correct.")
            
            # Create a unique filename based on current time
            video_counter += 1
            video_path = f'attendance_video_{video_counter}.avi'
            capture_video(video_path, duration=10)  # Capture video for 10 seconds

            # Mark attendance
            mark_attendance()

            # Display the captured video
            video_capture = cv2.VideoCapture(video_path)
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                cv2.imshow('Attendance Marked Video', frame)
                if cv2.waitKey(1) & 0xFF == 13:  # Check for Enter key press
                    break
            video_capture.release()
            cv2.destroyAllWindows()

        else:
            cv2.putText(frame, 'Uniform Incorrect', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            speak("Uniform is incorrect.")
        
        # Display the resulting frame
        cv2.imshow('Uniform Detection', frame)
        
        # Break loop on Enter key press
        if cv2.waitKey(1) & 0xFF == 13:  # Check for Enter key press
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
