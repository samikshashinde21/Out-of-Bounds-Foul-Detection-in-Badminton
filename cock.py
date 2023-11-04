import cv2

# Load the image
img = cv2.imread('D:/CV_CP/Badminton/frames_1/1.png')

# Display the image
cv2.imshow('image', img)

# Define a mouse callback function


def mouse_callback(event, x, y, flags, param):
    # If left button is clicked, print the coordinates of the clicked point
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'x: {x}, y: {y}')


# Set the mouse callback function for the image window
cv2.setMouseCallback('image', mouse_callback)

# Wait for user input
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()
