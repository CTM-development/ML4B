import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys

import matplotlib.pyplot as plt

# import the image
img_fldr_path = "/home/darwin/Data/wgisdt_dataset/data_raw"
img = Image.open(img_fldr_path + "/CDY_2015.jpg")

# import the corresponding txt file
img_grapeBoxes = []
with open(img_fldr_path + "/CDY_2015.txt", 'r') as f:
    img_grapeBoxes = f.readlines()

img_grapeBoxes_np = np.array(img_grapeBoxes)
img_grapeBoxes_np = img_grapeBoxes_np.reshape(img_grapeBoxes_np.shape[0], 1)
print(img_grapeBoxes_np.shape)

b = img_grapeBoxes
print(b)
b = [l.replace('\n', '').split(' ') for l in b]
for i, l in enumerate(b):
    print(f'line number {i} contains : {l}')

b_np = np.array(b)
print(b_np[0])

sys.exit()

# convert img to numpay array
img_np = np.array(img)
# print(img_np.shape)
# # display img
# plt.imshow(img_np)
# plt.xticks([])
# plt.yticks([])
# plt.show()
#

img_boxes_edgePoints = []

img_width = 2048
img_height = 1365

img_padding = 1

for x in img_grapeBoxes_np:
    s = x[0].replace('\n', '')
    x_data = s.split(' ')
    x_data = [float(x) for x in x_data]

    box_cX = x_data[1] * img_width
    box_cY = x_data[2] * img_height

    box_width = int(x_data[3]*img_width*img_padding)
    box_height = int(x_data[4]*img_height*img_padding)

    box_topLeft_x = np.clip(int(box_cX - 0.5 * box_width), 0, img_width)
    box_topLeft_y = np.clip(int(box_cY - 0.5 * box_height), 0, img_height)

    box_diagonalEdgePoints = np.array(
        [
            [box_topLeft_x, box_topLeft_y],
            [np.clip(box_topLeft_x+box_width, 0, img_width), np.clip(box_topLeft_y+box_height, 0, img_height)]
        ]
    )

    img_boxes_edgePoints.append(box_diagonalEdgePoints)

print(img_boxes_edgePoints[0])

img_clipped_grapes = np.array([])

for edgePoints in img_boxes_edgePoints:
    clipped_grape = img_np[edgePoints[0, 1]:edgePoints[1, 1], edgePoints[0, 0]:edgePoints[1, 0]]

    # print("gripped grapes: ")
    # print(clipped_grape.shape)
    # print(gripped_grape)

    img_clipped_grapes = np.append(img_clipped_grapes, clipped_grape)

    plt.imshow(clipped_grape)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# some testing
# print(img_np)
# print("img_np[0][0]: \n", img_np[0][0])
# print(img_np.shape)
#
#
# img_cut = img_np[0:1000, :1000]
# print("img_cut shape: \n", img_cut.shape)
#
#


# # plot the pixel values
# plt.hist(img_np.ravel(), bins=50, density=True)
# plt.xlabel("pixel values")
# plt.ylabel("relative frequency")
# plt.title("distribution of pixels")
# plt.show()
#
# # custom transformer function~
# transform = transforms.Compose(
#     [transforms.ToTensor()]
# )
#
# img_tr = transform(img)
# print(img_tr)
# print(img_tr.shape)
#
# # calculate mean and std
# mean, std = img_tr.mean([1, 2]), img_tr.std([1, 2])
#
# # print mean and std
# print("mean and std before normalize:")
# print("Mean of the image:", mean)
# print("Std of the image:", std)
#
# transform_norm = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)
# ])
#
# img_normalized = transform_norm(img)
#
# img_normalized_np = np.array(img_normalized)
#
# # plot the pixel values
# plt.hist(img_normalized_np.ravel(), bins=50, density=True)
# plt.xlabel("pixel values")
# plt.ylabel("relative frequency")
# plt.title("distribution of pixels")
# plt.show()
#
# img_normalized_np = img_normalized_np.transpose(1, 2, 0)
#
# #display normalized and raw image
# plt.imshow(img_np)
# plt.imshow(img_normalized_np)
# plt.xticks([])
# plt.yticks([])
# plt.show()
