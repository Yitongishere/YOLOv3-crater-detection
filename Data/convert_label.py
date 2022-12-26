"""
    This script combines all of the annotations files into one txt file and convert the format of labels from
    xywh to xyxy for training.
"""

import os
# new generated txt file with all of the labels
filename = './data_train.txt'
# the directory contains all original annotations file
file_dir = './Annotations_trainval'


def xywh_to_xyxy(xywh):
    """
    convert the format of labels from xywh to xyxy.

    Parameters
    ----------
    xywh: list [x, y, w, h], center point x, center point y, width and height respectively.

    Returns
    ----------
    xyxy: list [x1, y1, x2, y2], top-left-x, top-left-y, bottom-right-x, bottom-right-y.
    """
    xyxy = []
    xyxy.append(xywh[0] - xywh[2] // 2)
    xyxy.append(xywh[1] - xywh[3] // 2)
    xyxy.append(xywh[0] + xywh[2] // 2)
    xyxy.append(xywh[1] + xywh[3] // 2)

    return xyxy


with open(filename, 'w') as file_object:
    for root, dirs, files in os.walk(file_dir):
        files = [f for f in files if not f[0] == '.']
        name = files
        for i in name:
            with open(file_dir+'/'+i, "r") as f:
                data = f.readlines()  
                s = ' '
                for j in data:
                    clss = j[0]
                    j = j.split('\n')[0].split(' ')
                    j = list(map(lambda x: int(416*float(x)), j[1:]))
                    s += ','.join(str(q) for q in xywh_to_xyxy(j))
                    s += ','
                    s += clss
                    s += ' '
                i = i.split('.')[0]
                # the directory contains the training images
                file_object.write("./Data/data_images/")
                file_object.write(i + '.png')
                file_object.write(s[:-1])
                file_object.write('\n')


