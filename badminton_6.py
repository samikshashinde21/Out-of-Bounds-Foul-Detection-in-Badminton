import os
import cv2
import numpy as np
import pandas as pd
import re


pred = os.listdir("D:/CV_CP/Badminton/cock_pred/")
pred.sort(key=lambda f: int(re.sub('\D', '', f)))

df = pd.read_csv("coorinate.csv")
for i in pred:
    k = i.split('_')
    num1 = int(k[0])
    k = k[1].split('.')
    num2 = int(k[0])
    # print(f'{num1} and {num2}')
    points = df.loc[((df['framej'] == num1) & (df['patchi'] == num2)), [
        'framej', 'x', 'y', 'w', 'h']]
    img = cv2.imread("D:/CV_CP/Badminton/frames/" + str(num1)+".png")
    img = cv2.rectangle(img, (int(points['x']), int(points['y'])), (int(
        points['x'])+int(points['w']), int(points['y'])+int(points['h'])), (255, 0, 0), 2)

    cv2.imwrite("D:/CV_CP/Badminton/frames/" + str(num1) + ".png", img)


frames = os.listdir("D:/CV_CP/Badminton/frames/")
pred.sort(key=lambda f: int(re.sub('\D', '', f)))

for i in range(len(frames)):
    img = cv2.imread("D:/CV_CP/Badminton/frames/" + str(i+1) + ".png")
    cv2.imshow("img", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
