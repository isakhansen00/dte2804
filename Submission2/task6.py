import cv2
import numpy as np
# Create a VideoCapture object to read the video file
cap = cv2.VideoCapture("video001_kort.mp4")

tracker = cv2.TrackerCSRT_create()

# Read the first frame to initialize the tracker
ret, frame = cap.read()
roi = cv2.selectROI(frame, False)
tracker.init(frame, roi)

# Loop through the frames and perform tracking
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update the tracker
    success, roi = tracker.update(frame)
    
    # Draw the bounding box
    if success:
        (x, y, w, h) = tuple(map(int, roi))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Object Tracking', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1000//30) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()