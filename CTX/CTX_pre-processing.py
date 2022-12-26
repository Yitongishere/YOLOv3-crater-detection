"""
    This script is designed for combining multiple processing steps for CTX images into one.
    By running this script, resizing, stacking, cropping, ground-truth labels generating and
    visualizing will be done in right order.
"""

import os
import argparse
from CTX_utils import resize_img, stack_img, crop_img, visualize_map, map_robbins_label

def arg_parse():
    """
    Parse arguments to process the CTX images with our data processing pipeline
    """
    parser = argparse.ArgumentParser(description='CTX images processing pipeline')
    parser.add_argument("--resized_size", type=int, default=1248, help="the size of resized CTX images")
    parser.add_argument("--min", type=float, default=1.0,
                        help="the size of the smallest crater to be mapped from Robbins dataset")
    parser.add_argument("--max", type=float, default=32.0,
                        help="the size of the largest crater to be mapped from Robbins dataset")

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    # config of CTX images processing, setting below parameters properly before running
    size_of_resized_img = args.resized_size
    latitude_range = [-20, -14]
    longitude_range = [174, 180]
    crater_size_range = [args.min, args.max]

    # create the directories for saving resized images and cropped images if they does not exist already
    if not os.path.exists('./resized_CTX'):
        os.makedirs('./resized_CTX')
    if not os.path.exists('./cropped_images'):
        os.makedirs('./cropped_images')

    # remove pre-existing files in the directory
    ls = os.listdir('./cropped_images/')
    for i in ls:
        os.remove('./cropped_images/' + i)
    ls = os.listdir('./resized_CTX/')
    for i in ls:
        os.remove('./resized_CTX/' + i)

    # process the original CTX images according to the data pipeline
    ls = os.listdir('./original_CTX/')
    print("Resizing...")
    for i in ls:
        if i.endswith('.tif'):
            resize_img(i, size_of_resized_img)
    print("Stacking...")
    stack_img()
    print("Cropping...")
    crop_img('./stacked_CTX.png')

    # get the ground truth label for evaluating the performance
    map_robbins_label(size_of_resized_img*3, latitude_range, longitude_range, crater_size_range)

    # visualize the ground-truth craters on panoramic view
    map_robbins_label(1, latitude_range, longitude_range, crater_size_range, "./visualization.txt")
    visualize_map("stacked_CTX.png")