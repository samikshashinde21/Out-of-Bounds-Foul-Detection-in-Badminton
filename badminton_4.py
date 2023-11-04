import cv2
import numpy as np
import os
import re

orb = cv2.ORB_create()

path = "D:/CV_CP/Badminton/shuttlecock_1/cock/"
frames = os.listdir(path)
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

images = []
className = []
# print(len(frames))
for i in frames:
    img_cur = cv2.imread(
        "D:/CV_CP/Badminton/shuttlecock_1/cock/"+i, 0)
    images.append(img_cur)

print(len(images))
cv2.imshow('img22', images[2])
cv2.waitKey()


def findes(images):
    deslist = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        print(kp, des)
        print("\n")
        deslist.append(des)
    return deslist


sumall = []
matchlist = []
good = []


def finid(img, deslist, p2, p3):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    for des in deslist:
        matches = bf.knnMatch(des, des2, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        matchlist.append(len(good))
    
deslist = findes(images)
print(f'the descripters {(deslist)}')

images2 = []
path2 = "D:/CV_CP/Badminton/patches/"
frame1 = os.listdir(path2)
for j in frame1:
    img2_cur = cv2.imread(f'{path2}/{j}', 0)
    p1 = j.split('_')
    p2 = int(p1[0])
    p3 = p1[1].split('.')
    p3 = int(p3[0])
   
