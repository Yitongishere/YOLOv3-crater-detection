"""
    Use K-means clustering to find optimal anchors sizes for our selected THEMIS dataset.
    The optimal anchors sizes will write into 'anchor.txt'

    referenced and modified from:
    https://gitee.com/benjiaxu/yolov4-keras/blob/master/kmeans_for_anchors.py with MIT License
    different method for loading data was used in order to match our annotations
"""

import numpy as np


def load_data(path):
    """
    calculate the width and height of each craters in our annotations from txt file

    Parameters
    ----------
    path: string, path of data annotations

    Returns
    ----------
    np.array(data): 2-d np.array, (number of craters, 2) in shape,
                    representing normalized width and height of the crater
    """
    data = []
    height = 416.
    width = 416.
    f = open(path)
    for lines in f:
        if len(lines) > 0:
            items = lines.split(' ')[1:]

            # get width and height of each crater
            for item in items:
                xmin = int(item.split(',')[0])
                ymin = int(item.split(',')[1])
                xmax = int(item.split(',')[2])
                ymax = int(item.split(',')[3])

                xmin = xmin / width
                ymin = ymin / height
                xmax = xmax / width
                ymax = ymax / height

                data.append([xmax - xmin, ymax - ymin])

    return np.array(data)


def cas_iou(box, cluster):
    """
    Calculating IoU between a bounding box and 9 clusters with same center point

    Parameters
    ----------
    box: np.array, [w, h], width and height
    cluster: np.array

    Returns
    ----------
    iou: np.array, IoU between the box and each of the cluster
    """
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    """
    Calculating average IoU of every ground-truth bbox with its corresponding anchor
    """
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    """
    Finding the optimal sizes of anchor boxes for our dataset

    Parameters
    ----------
    box: np.array, (number of craters, 2) in shape
    k: number of anchors boxes, 9 for yolov3

    Returns
    ----------
    cluster: np.array, (9, 2) in shape. Clusters after applying k-means clustering
    """
    # number of craters we have in annotations
    row = box.shape[0]
    # The position of each box
    distance = np.empty((row, k))
    # last position of clusters
    last_clu = np.zeros((row,))
    # Randomly select 9 cluster centers
    # different seed will lead different result, we select the seed (28) with the highest accuracy
    np.random.seed(28)
    cluster = box[np.random.choice(row, k, replace=False)]
    while True:
        # Calculate the iou of 9 clusters and each craters
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        # classify each crater to the nearest cluster
        near = np.argmin(distance, axis=1)
        if (last_clu == near).all():
            break
        # Find the center of each class
        for j in range(k):
            cluster[j] = np.median(
                box[near == j], axis=0)
        last_clu = near
    return cluster


SIZE = 416
anchors_num = 9
data = load_data('./label.txt')
print(data)

out = kmeans(data, anchors_num)
out = out[np.argsort(out[:, 0])]
result = out * SIZE
print(result)
# Average IoU of every ground-truth bbox with its corresponding anchor
print('Accuracy: {:.2f}%'.format(
    avg_iou(data, out) * 100))
# write into a txt file
f = open("anchors.txt", 'w')
row = np.shape(result)[0]
for i in range(row):
    x_y = "%d,%d\n" % (round(result[i][0]), round(result[i][1]))
    f.write(x_y)
f.close()
