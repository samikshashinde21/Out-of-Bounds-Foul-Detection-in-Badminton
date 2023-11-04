import os
import re
import cv2
# import imutils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

folders = os.listdir("D:/CV_CP/Badminton/shuttlecock/")

images = []
labels = []
for folder in folders:
    files = os.listdir(
        'D:/CV_CP/Badminton/shuttlecock/' + folder)
    label = folders.index(folder)
    for file in files:
        img = cv2.imread(
            'D:/CV_CP/Badminton/shuttlecock/' + folder + '/' + file, 0)
        img = cv2.resize(img, (25, 25))

        images.append(img)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)
features = images.reshape(len(images), -1)
np.save('labelcock.npy', labels)
np.save('features.npy', images)

x_tr, x_val, y_tr, y_val = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=0)

rfc = RandomForestClassifier(max_depth=3)
rfc.fit(x_tr, y_tr)

y_pred = rfc.predict(x_val)
# print(classification_report(y_val, y_pred))

cock_df = pd.DataFrame(columns=['frame', 'x', 'y', 'w', 'h'])

# listing down all the file names
frames = os.listdir('D:/CV_CP/Badminton/frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

for idx in range(len(frames)):

    img = cv2.imread(
        'D:/CV_CP/Badminton/frames/' + frames[idx])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, image = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # !rm - r
    # patch / *

    num = 20
    cnt = 0

    df = pd.DataFrame(columns=['frame', 'x', 'y', 'w', 'h'])
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        # print(x)
        numer = min([w, h])
        denom = max([w, h])
        ratio = numer / denom

        if (x >= num and y >= num):
            xmin, ymin = x - num, y - num
            xmax, ymax = x + w + num, y + h + num
        else:
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h

        if (ratio >= 0.5):
            # print(cnt,x,y,w,h,ratio)
            df.loc[cnt, 'frame'] = frames[idx]
            df.loc[cnt, 'x'] = x
            df.loc[cnt, 'y'] = y
            df.loc[cnt, 'w'] = w
            df.loc[cnt, 'h'] = h

            cv2.imwrite("D:/CV_CP/Badminton/patches/" +
                        str(cnt) + ".png", img[ymin:ymax, xmin:xmax])
            cnt = cnt + 1

    files = os.listdir("D:/CV_CP/Badminton/patches/")
    if (len(files) > 0):

        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        test = []
        for file in files:
            img = cv2.imread(
                'D:/CV_CP/Badminton/patches/' + file, 0)
            img = cv2.resize(img, (25, 25))
            test.append(img)

        test = np.array(test)

        test = test.reshape(len(test), -1)
        y_pred = rfc.predict(test)
        prob = rfc.predict_proba(test)

        if 0 in y_pred:
            ind = np.where(y_pred == 0)[0]
            proba = prob[:, 0]
            confidence = proba[ind]
            confidence = [i for i in confidence if i > 0.7]
            if (len(confidence) > 0):

                maximum = max(confidence)
                cock_file = files[list(proba).index(maximum)]

                img = cv2.imread(
                    'D:/CV_CP/Badminton/patches/' + cock_file)
                cv2.imwrite(
                    'D:/CV_CP/Badminton/shuttlecock/' + str(frames[idx]), img)

                no = int(cock_file.split(".")[0])
                # print()
                cock_df.loc[idx, 'frame'] = frames[idx]
                cock_df.loc[idx, 'x'] = x
                cock_df.loc[idx, 'y'] = y
                cock_df.loc[idx, 'w'] = w
                cock_df.loc[idx, 'h'] = h

            else:
                cock_df.loc[idx, 'frame'] = frames[idx]

        else:
            cock_df.loc[idx, 'frame'] = frames[idx]

cock_df.dropna(inplace=True)
print(cock_df)

files = cock_df['frame'].values

num = 10
for idx in range(len(files)):
    # draw contours
    img = cv2.imread('D:/CV_CP/Badminton/frames/' + files[idx])
    x = cock_df.loc[idx, 'x']
    y = cock_df.loc[idx, 'y']
    w = cock_df.loc[idx, 'w']
    h = cock_df.loc[idx, 'h']
    print(xmin, ymin, xmax, ymax)
    xmin = x - num
    ymin = y - num
    xmax = x + w + num
    ymax = y + h + num

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.imwrite(
        "D:/CV_CP/Badminton/frames/" + files[idx], img)

    frames = os.listdir('D:/CV_CP/Badminton/frames/')
    frames.sort(key=lambda f: int(re.sub('\D', '', f)))

    frame_array = []

    for i in range(len(frames)):
        # reading each files
        img2 = cv2.imread(
            'D:/CV_CP/Badminton/frames/' + frames[i])
        height, width, layers = img.shape
        size = (width, height)
        # inserting the frames into an image array
        # cv2.imshow('final',img)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        frame_array.append(img2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        'D:/CV_CP/Badminton/frames/badminton001.mp4', fourcc, 25, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
