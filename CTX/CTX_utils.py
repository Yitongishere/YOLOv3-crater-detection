"""
    This script includes the functions that used in our data processing pipeline.
"""

from PIL import Image
import cv2
import os
import numpy as np


def resize_img(filename, new_size=1248):
    """
    Resizing the image. The resized image is automatically saved in the './resized_CTX/' directory.

    Parameters
    ----------
    filename: string, path of the image to be resized
    new_size: int, size of resized image (pixel)
    """
    img = cv2.imread('original_CTX/' + filename, -1)

    # resize image
    size = (new_size, new_size)
    shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    new_file = 'resized_CTX/' + filename[:-4] + '_' + str(new_size) +'.png'
    cv2.imwrite(new_file, shrink)

    return


def stack_img():
    """
    Stacking the 9 resized CTX images into a panoramic view.
    The stacked image is automatically saved as './stacked_CTX.png'
    """
    dir = './resized_CTX/'
    files = os.listdir(dir)

    for file in files:
        if file.endswith('.tif') or file.endswith('.png'):

            # ----------------------------------------------------------------

            if int(file[3]) == 0 and int(file[4]) == 0:
                image_0_0 = cv2.imread(dir + file)

            elif int(file[3]) == 0 and int(file[4]) == 1:
                image_0_1 = cv2.imread(dir + file)

            elif int(file[3]) == 0 and int(file[4]) == 2:
                image_0_2 = cv2.imread(dir + file)

            elif int(file[3]) == 1 and int(file[4]) == 0:
                image_1_0 = cv2.imread(dir + file)

            elif int(file[3]) == 1 and int(file[4]) == 1:
                image_1_1 = cv2.imread(dir + file)

            elif int(file[3]) == 1 and int(file[4]) == 2:
                image_1_2 = cv2.imread(dir + file)

            elif int(file[3]) == 2 and int(file[4]) == 0:
                image_2_0 = cv2.imread(dir + file)

            elif int(file[3]) == 2 and int(file[4]) == 1:
                image_2_1 = cv2.imread(dir + file)

            elif int(file[3]) == 2 and int(file[4]) == 2:
                image_2_2 = cv2.imread(dir + file)

            # ----------------------------------------------------------------


    image_0 = np.hstack([image_0_0, image_0_1, image_0_2])
    image_1 = np.hstack([image_1_0, image_1_1, image_1_2])
    image_2 = np.hstack([image_2_0, image_2_1, image_2_2])

    image = np.vstack([image_0, image_1, image_2])
    cv2.imwrite('./stacked_CTX.png', image)

    return


def crop_img(filename='./stacked_CTX.png'):
    """
    Cropping the stacked image into tiles with each of tiles in the size of 416 by 416 pixels.
    The cropped tiles are stored in the './cropped_images/' directory for further crater detecting.

    Parameters
    ----------
    filename: string, path of the stacked CTX image (panoramic image)
    """
    img = Image.open(filename)
    ori_size = img.size[0]
    piece_size = 416

    # remove pre-existing files in the directory
    ls = os.listdir('./cropped_images/')
    for i in ls:
        os.remove('./cropped_images/' + i)

    for i in range(int(ori_size/piece_size)):
        for j in range(int(ori_size/piece_size)):
            box = (piece_size * j, piece_size * i, piece_size * (j + 1), piece_size * (i + 1))
            region = img.crop(box)
            if i > 9 and j > 9:
                region.save('./cropped_images/{}_{}_{}.png'.format(filename[:-4], i, j))
            elif i < 9 and j < 9:
                region.save('./cropped_images/{}_0{}_0{}.png'.format(filename[:-4], i, j))
            elif i < 9 and j > 9:
                region.save('./cropped_images/{}_0{}_{}.png'.format(filename[:-4], i, j))
            elif i > 9 and j < 9:
                region.save('./cropped_images/{}_{}_0{}.png'.format(filename[:-4], i, j))

    return

def visualize_map(filename):
    """
    Drawing the ground-truth bounding boxes on the panoramic image

    Parameters
    ----------
    filename: string, path of the stacked CTX image (panoramic image)
    """
    image = cv2.imread(filename)
    h, w = image.shape[:2]
    with open("./visualization.txt", 'r') as file:
        lines = file.readlines()

        for line in lines:
            x1 = float(line.split(' ')[0]) * w
            y1 = float(line.split(' ')[1]) * h
            x2 = float(line.split(' ')[2]) * w
            y2 = float(line.split(' ')[3]) * h

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

    new_file = './gt_' + filename
    cv2.imwrite(new_file, image)

    return

def map_robbins_label(img_size, latitude_range, longitude_range, size_range, new_file="./selected_craters_XY.txt"):
    """
    Mapping the ground-truth labels from Robbins crater dataset to CTX images. By specify the range of
    latitude and longitude, the craters in this region with specified size range will be picked out from
    Robbins crater dataset and write into a txt file.

    Parameters
    ----------
    img_size: int, size of panoramic imaage
    latitude_range: list [a,b], a list of latitude_range, only craters in this latitude range are picked out
    longitude_range: list [a,b], a list of longitude_range, only craters in this longitude range are picked out
    size_range: list [a,b], a list of the size range, only craters in this size range are picked out
    new_file: string, the path of the new created txt file, which save the coordinates of picked out craters.
            with 4 entries for each craters, top-left-x, top-left-y, bottom-right-x, bottom-right-y
    """
    original_file = "./RobbinsCraters_tab.txt"
    f_new = open(new_file, 'w')
    f_ori = open(original_file, 'r')

    line = f_ori.readline()
    # Skip the first line (Header)
    line = f_ori.readline()

    while line:
        a = line.split()
        if (float(a[1]) >= latitude_range[0] and float(a[1]) <= latitude_range[1]) \
                and (float(a[2]) >= longitude_range[0] and float(a[2]) <= longitude_range[1]) \
                and (float(a[5]) >= size_range[0] and float(a[5]) <= size_range[1]):

            a[1] = img_size - (float(a[1]) - latitude_range[0]) * img_size / (latitude_range[1] - latitude_range[0])
            a[2] = (float(a[2]) - longitude_range[0]) * img_size / (longitude_range[1] - longitude_range[0])
            # 1000: km-m,  5: resolution of CTX, 23710: original size of CTX, 3: image are stacke by 3x3 resized CTX
            a[5] = float(a[5]) * 1000 * img_size / (5*23710*3)

            y1 = str(float(a[1] - a[5]/2))
            x1 = str(float(a[2] - a[5]/2))
            y2 = str(float(a[1] + a[5]/2))
            x2 = str(float(a[2] + a[5]/2))
            f_new.write(x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + '\n')

        line = f_ori.readline()

    f_new.close()
    f_ori.close()

    return



