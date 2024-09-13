import cv2
import numpy as np
import pyttsx3
import time

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

def main():
    """Main function to detect uniform and logo."""
    # Path to the reference logo image
    logo_image_path = r'C:\Users\Hey!\OneDrive\Desktop\Tuesday\uniform_logo.jpg'
    
    # Define uniform color in HSV space (Blue color range)
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

    start_time = time.time()
    detected = False

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
            detected = True
            cv2.putText(frame, 'Uniform Correct', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            speak("Uniform is correct.")
        else:
            cv2.putText(frame, 'Uniform Incorrect', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            speak("Uniform is incorrect.")

        # Display the frame with the result
        cv2.imshow('Uniform Detection', frame)
        
        # Stop the script if the uniform is detected
        if detected:
            break

        # Check if 10 seconds have passed
        if time.time() - start_time > 10:
            break  # Timeout after 10 seconds

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
