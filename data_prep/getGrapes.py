from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import skimage.transform as ski_trans


def read_pair_dict(dir_path, print_process: bool = False):
    dir_content = os.scandir(dir_path)

    # strip file names off their extension and delete duplicates
    entry_names = [str(entry.name).replace('.txt', '').replace('.jpg', '') for entry in dir_content]
    entry_names = list(dict.fromkeys(entry_names))

    file_count = 0
    if print_process:
        file_count = entry_names.__len__()

    imgs_and_boxes = []

    for i, name in enumerate(entry_names):
        if print_process:
            if i % 10 == 0:
                print(f'File processing: {i}/{file_count}')

        # read jp and convert to nparray
        img = Image.open(dir_path + "/" + name + '.jpg')
        img_np = np.array(img)

        # read txt and convert to ndarray of shape=x, 1 where x = lines in txt file
        with open(dir_path + '/' + name + '.txt', 'r') as f:
            boxes = f.readlines()

        # replace endofline and split strinf into list then convert to nparray
        boxes = [line.replace('\n', '').split(' ') for line in boxes]

        # typecasting line elements from string to float
        boxes = [list(map(float, line)) for line in boxes]

        boxes_np = np.array(boxes)

        # create img and box tupel and add to re
        img_and_box = [img_np, boxes_np]
        imgs_and_boxes.append(img_and_box)

    return imgs_and_boxes


def cut_out_grapes(img: np.ndarray, box_lines: np.ndarray, img_padding: float = 1, img_squared: bool = False):
    img_boxes_edgePoints = []

    img_width = 2048
    img_height = 1365

    for line in box_lines:
        box_cX = line[1] * img_width
        box_cY = line[2] * img_height

        box_width = int(line[3] * img_width * img_padding)
        box_height = int(line[4] * img_height * img_padding)

        if img_squared:
            box_width = max(box_height, box_width)
            box_height = box_width

        box_topLeft_x = np.clip(int(box_cX - 0.5 * box_width), 0, img_width)
        box_topLeft_y = np.clip(int(box_cY - 0.5 * box_height), 0, img_height)

        box_diagonalEdgePoints = np.array(
            [
                [box_topLeft_x, box_topLeft_y],
                [np.clip(box_topLeft_x + box_width, 0, img_width), np.clip(box_topLeft_y + box_height, 0, img_height)]
            ]
        )

        img_boxes_edgePoints.append(box_diagonalEdgePoints)

    img_clipped_grapes = []

    for edgePoints in img_boxes_edgePoints:
        clipped_grape = np.array(img[edgePoints[0, 1]:edgePoints[1, 1], edgePoints[0, 0]:edgePoints[1, 0]])

        img_clipped_grapes.append(clipped_grape)

    return img_clipped_grapes


# reading in all the pictures and txt files from img_folder_path (bunch of sad hardcoding is done here :| )
img_folder_path = "/home/darwin/Data/wgisdt_dataset/data_raw"

grapes = list()

# call read_pair_dict to read in all the unique names from specifies folder excluding extensions .txt and .jpg
imgs_and_boxes = read_pair_dict(img_folder_path, print_process=False)

#
for i, t in enumerate(imgs_and_boxes):
    if i % 10 == 0:
        print(f'Cutting grapes: {i}/{imgs_and_boxes.__len__()}')
    grapes.append(cut_out_grapes(img=t[0], box_lines=t[1], img_padding=1, img_squared=True))

# flattening the list "grapes"
grapes = [val for sublist in grapes for val in sublist]
grapes = np.array(grapes)
print(grapes.shape)

res = 128

# save the grapes data:
for i, g in enumerate(grapes):
    # resizing (comment out if no resizing is needed)
    g = ski_trans.resize(g, (res, res), preserve_range=True, anti_aliasing=True)

    print(f'writing files: {i+1}/{grapes.__len__()}')
    np.save("/home/darwin/Data/wgisdt_dataset/data_grapes_squared_cut_out_padding0_128x128/as_npy/grape_"+str(i), g)
    # save as .png
    if i % 100 == 0:
        g = Image.fromarray((g).astype(np.uint8))
        g.save("/home/darwin/Data/wgisdt_dataset/data_grapes_squared_cut_out_padding0_128x128/as_jpg/grape_"+str(i)+".jpg")
