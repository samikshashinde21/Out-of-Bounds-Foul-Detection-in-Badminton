import csv
import cv2
import numpy as np

# Define the vertices of the badminton court
court_vertices = np.array([[51, 57], [0, 721], [1064, 721], [926, 37]])


# Get input coordinates of the shuttlecock from the user
x = int(input("Enter x-coordinate of shuttlecock: "))
y = int(input("Enter y-coordinate of shuttlecock: "))
shuttlecock_coords = (x, y)

# Check if the shuttlecock is within the court
if cv2.pointPolygonTest(court_vertices, shuttlecock_coords, False) == 1:
    print("IN")
else:
    print("OUT")
