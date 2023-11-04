import cv2
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import Augmentor
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from skimage import feature
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 6
path = r"D:/CV_CP/Badminton/shuttlecock_1/cock"
filenames = os.listdir(path)
filenames.sort()
random.seed(230)
random.shuffle(filenames)
split_1 = int(0.15 * len(filenames))
cock_test_filenames = filenames[:split_1]
cock_train_filenames = filenames[split_1:]

print(len(cock_train_filenames), len(cock_test_filenames))

# 7 cock
images0 = []
labels0 = []

for i in range(len(cock_train_filenames)):
    img = cv2.imread(
        path + '\\' + cock_train_filenames[i], cv2.IMREAD_GRAYSCALE)
    re_img = cv2.resize(img, (64, 128))
    images0.append(re_img)
    labels0.append('B')

test_images0 = []
test_labels0 = []

for i in range(len(cock_test_filenames)):
    img = cv2.imread(
        path + '\\' + cock_test_filenames[i], cv2.IMREAD_GRAYSCALE)
    re_img = cv2.resize(img, (64, 128))
    test_images0.append(re_img)
    test_labels0.append('B')

path = r"D:/CV_CP/Badminton/shuttlecock_1/no cock"
filenames = os.listdir(path)
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
# shuffles the ordering of filenames (deterministic given the chosen seed)
random.shuffle(filenames)
split_1 = int(0.15 * len(filenames))
no_cock_test_filenames = filenames[:split_1]
no_cock_train_filenames = filenames[split_1:]

print(len(no_cock_train_filenames), len(no_cock_test_filenames))

# No_cock
images1 = []
labels1 = []

for i in range(len(no_cock_train_filenames)):
    img = cv2.imread(
        path + '\\' + no_cock_train_filenames[i], cv2.IMREAD_GRAYSCALE)
    re_img = cv2.resize(img, (64, 128))
    images1.append(re_img)
    labels1.append('N')


test_images1 = []
test_labels1 = []

for i in range(len(no_cock_test_filenames)):
    img = cv2.imread(
        path + '\\' + no_cock_test_filenames[i], cv2.IMREAD_GRAYSCALE)
    re_img = cv2.resize(img, (64, 128))
    test_images1.append(re_img)
    test_labels1.append('N')

labels = labels0 + labels1
images = images0 + images1

hog = []
for image in images:
    hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(
        32, 32), cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    hog.append(hog_desc)

df_hog = pd.DataFrame(hog, dtype=object)

df_hog['Labels'] = labels

print(df_hog)

svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(df_hog.iloc[:, 0:108], df_hog['Labels'])


print(svm_model)

test_images = test_images0 + test_images1
test_labels = test_labels0 + test_labels1
print(len(test_images))
# print((test_labels))

test_labels[22]

# cv2.imshow("IMG22",test_images[23])
# cv2.waitKey(0)

patch_images = []
patch_name = []
patch_path = r"D:/CV_CP/Badminton/patches"
filenames = os.listdir(patch_path)
for i in range(len(filenames)):
    img = cv2.imread(patch_path + '/' + filenames[i], cv2.IMREAD_GRAYSCALE)
    re_img = cv2.resize(img, (64, 128))
    patch_images.append(re_img)
    patch_name.append(filenames[i])
    # print(i)

print("success1")

patch_prediction = []
patch_image_of_hog = []
# patch_hog_test = []
for i in range(len(patch_images)):
    (hog_desc, hog_image) = feature.hog(patch_images[i], orientations=9, pixels_per_cell=(
        32, 32), cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)
    # hog_test.append(hog_desc)
    patch_pred = svm_model.predict(hog_desc.reshape(1, -1))[0]
    hog_image = hog_image.astype('float64')
    patch_prediction.append(patch_pred)
    patch_image_of_hog.append(hog_image)

print(patch_prediction)

cock_patch = []
for i in range(len(patch_prediction)):
    if patch_prediction[i] == 'B':
        cock_patch.append(patch_name[i])
        cv2.imwrite(
            f"D:/CV_CP/Badminton/cock_pred/{patch_name[i]}", patch_images[i])

print(len(cock_patch))
