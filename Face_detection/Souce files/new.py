import cv2
import numpy as np

# Load the template logo image in grayscale
template = cv2.imread(r'C:\Users\Hey!\OneDrive\Desktop\Tuesday\uniform_logo.jpg', 0)
template_height, template_width = template.shape

# Initialize video capture and video writer
cap = cv2.VideoCapture(0)  # 0 for default camera
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

def detect_logo(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply template matching
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    
    # Set a threshold to consider it a match
    threshold = 0.8
    loc = np.where(result >= threshold)
    
    # If any region meets the threshold, consider the logo detected
    if len(loc[0]) > 0:
        return True
    
    return False

recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the logo in the frame
    if detect_logo(frame):
        if not recording:
            print("Present - Starting Video Recording")
            recording = True
        out.write(frame)
    else:
        if recording:
            print("Absent - Stopping Video Recording")
            recording = False

    # Display the frame (for debugging purposes)
    cv2.imshow('Frame', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
