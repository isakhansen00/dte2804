
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

# Function to apply selected algorithm to the video frames
def apply_algorithm():
    algorithm = algorithm_choice.get()

    if algorithm == "CSRT Tracker":
        tracker = cv2.TrackerCSRT_create()
        ret, frame = cap.read()
        roi = cv2.selectROI(frame, False)
        tracker.init(frame, roi)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            success, roi = tracker.update(frame)
            # Draw the bounding box
            if success:
                (x, y, w, h) = tuple(map(int, roi))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failed!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow('Video with ' + algorithm, frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1000//60) & 0xFF == ord('q'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if algorithm == "Hough Lines":
            # Apply Hough Line Transform
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Draw lines in red
        
        elif algorithm == "Otsu Thresholding":
            # Apply Otsu Thresholding
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            frame = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
        
        elif algorithm == "SIFT":
            # Apply SIFT algorithm
            sift = cv2.SIFT_create()
            kp, descriptors = sift.detectAndCompute(frame, None)
            frame = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        elif algorithm == "ORB":
            # Apply SIFT algorithm
            orb = cv2.ORB_create()
            kp = orb.detect(frame, None)
            kp, descriptors = orb.compute(frame, kp)
            frame = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        
        cv2.imshow('Video with ' + algorithm, frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1000//60) & 0xFF == ord('q'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
    
    cv2.waitKey(10)

# Create a VideoCapture object to read the video file
cap = cv2.VideoCapture("video001_kort.mp4")

# Create a Tkinter window
root = tk.Tk()
root.title("Video Processing")
root.geometry("300x100")
# Dropdown menu to select the algorithm
algorithm_choice = tk.StringVar()
algorithm_choice.set("Hough Lines")  # Default algorithm
algorithm_menu = ttk.Combobox(root, textvariable=algorithm_choice, values=["Hough Lines", "Otsu Thresholding", "SIFT", "ORB", "CSRT Tracker"])
algorithm_menu.pack(pady=10)

# Button to apply the selected algorithm
apply_button = tk.Button(root, text="Apply Algorithm", command=apply_algorithm)
apply_button.pack(pady=10)

# Start the Tkinter main loop
root.mainloop()

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
