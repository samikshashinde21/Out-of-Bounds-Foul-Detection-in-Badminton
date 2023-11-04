import cv2
import os

# Create a directory to store the frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Open the video file
cap = cv2.VideoCapture('D:/CV_CP/Badminton/dataset/video_10_A.mp4')

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error opening video file")
    # Initialize the frame counter
frame_count = 0

# Set ROI coordinates
x, y, w, h = 400, 300, 1100, 730

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Extract the ROI
    roi = frame[y:y+h, x:x+w]

    # Save the ROI as an image file
    filename = os.path.join('frames', f'{frame_count:04d}.jpg')
    cv2.imwrite(filename, roi)

    # Increment the frame counter
    frame_count += 1

# Release the video capture object
cap.release()
