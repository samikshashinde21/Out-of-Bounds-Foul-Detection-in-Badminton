import cv2
import numpy as np

# Load the image
img = cv2.imread('D:/CV_CP/Badminton/frames_1/1.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find the contours in the image
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the contour with the largest area
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# Approximate the contour to reduce the number of points
epsilon = 0.03 * cv2.arcLength(max_contour, True)
approx = cv2.approxPolyDP(max_contour, epsilon, True)

# Draw the contour on the image
cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)

# Get the coordinates of the vertices of the polygon
vertices = []
for vertex in approx:
    x, y = vertex[0]
    vertices.append((x, y))

# Display the image
cv2.imshow('Badminton Court', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the coordinates of the vertices of the polygon
print('Badminton Court Coordinates:')
for vertex in vertices:
    print('({}, {})'.format(vertex[0], vertex[1]))
