import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import re
import pandas as pd

# listing down all the file names
frames = os.listdir('D:/CV_CP/Badminton/frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

num = 25
low = 6
high = 31
cnt = 1

cock_df = pd.DataFrame(
    columns=['framej', 'patchi', 'x', 'y', 'w', 'h', 'lengthCon'])

for j in frames:

    img = cv2.imread('D:/CV_CP/Badminton/frames/' + j)
    j = j.split('.')
    j = int(j[0000])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 7, 150, 150)
    _, mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    contours, image = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, )
    if (j == 11):
        low = 11
        high = 20
    elif (j == 22):
        low = 17
        high = 31
    elif (j == 34):
        low = 7
        high = 18
    elif (j == 63):
        low = 6
        high = 18
    elif (j == 128):
        low = 7
        high = 21
    for i in range(len(contours)):
        if (len(contours[i]) <= high):
            if (len(contours[i]) >= low):
                x, y, w, h = cv2.boundingRect(contours[i])
                # Set ROI coordinates

                numer = min([w, h])
                denom = max([w, h])
                ratio = numer / denom

                if (x >= num and y >= num):
                    xmin, ymin = x - num, y - num
                    xmax, ymax = x + w + num, y + h + num
                else:
                    xmin, ymin = x, y
                    xmax, ymax = x + w, y + h

                cock_df.loc[cnt, 'framej'] = j
                cock_df.loc[cnt, 'patchi'] = i
                cock_df.loc[cnt, 'x'] = x
                cock_df.loc[cnt, 'y'] = y
                cock_df.loc[cnt, 'w'] = w
                cock_df.loc[cnt, 'h'] = h
                cock_df.loc[cnt, 'lengthCon'] = len(contours[i])
                print(cnt, j, i, x, y, w, h, ratio, len(contours[i]), i)
                cv2.imwrite("D:/CV_CP/Badminton/patches/" + str(j) + "_" + str(i) + ".png",
                            img[ymin:ymax, xmin:xmax])
                cnt = cnt + 1

                # if(ratio>=0.5 and ((w<=10) and (h<=10)) ):
#


print(cock_df)
cock_df.to_csv("coorinate.csv")
