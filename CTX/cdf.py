"""
    Use this script to generate the cumulative size-number distributions for variant image resolutions
"""

import matplotlib.pyplot as plt
import numpy as np

# Only for small craters (< 4km)
size_limit = 4

# data from Robbins dataset
f_t = open("./pred_results/target_size.txt", 'r')
line = f_t.readline()
cdf_t = np.array([])
a = line.split()
for i in range(len(a)):
    cdf_t = np.append(cdf_t, float(a[i]))
f_t.close()
print("Number of craters in Robbins dataset:", len(cdf_t))

# sort the data:
data_sorted_t = sorted(cdf_t, reverse=True)

# calculate the number of samples
p_t = 1. * np.arange(len(cdf_t)) / (len(cdf_t) - 1) * len(cdf_t)

# plot the sorted data:
ax1 = plt.plot(data_sorted_t, p_t)
ax1 = plt.xlabel('$Diameter (km)$')
ax1 = plt.ylabel('$Num$')
ax1 = plt.title('Cumulative distribution of Size (0-4 km)')

ax1 = plt.xscale('log')
ax1 = plt.yscale('log')

# 416 x 416
###########################################################################################

f_p = open("./pred_results/predictions_416.txt", 'r')
cdf_p = np.array([])
line = f_p.readline()
while line:
    a = line.split()
    size = ((float(a[2])-float(a[0])) + (float(a[3])-float(a[1]))) / 2 * 0.285
    if size < size_limit:
        cdf_p = np.append(cdf_p, size)

    line = f_p.readline()

f_p.close()
print("Number of detections on 416x416 images: ", len(cdf_p))

data_sorted_p_416 = sorted(cdf_p, reverse=True)
p_p_416 = 1. * np.arange(len(cdf_p)) / (len(cdf_p) - 1) * len(cdf_p)

# plot the sorted data:
ax1 = plt.plot(data_sorted_p_416, p_p_416)

# 832 x 832
###########################################################################################

f_p = open("./pred_results/predictions_832.txt", 'r')
cdf_p = np.array([])
line = f_p.readline()
while line:
    a = line.split()
    size = ((float(a[2])-float(a[0])) + (float(a[3])-float(a[1]))) / 2 * 0.142
    if size < size_limit:
        cdf_p = np.append(cdf_p, size)

    line = f_p.readline()

f_p.close()
print("Number of detections on 832x832 images: ", len(cdf_p))

data_sorted_p_832 = sorted(cdf_p, reverse=True)
p_p_832 = 1. * np.arange(len(cdf_p)) / (len(cdf_p) - 1) * len(cdf_p)

# plot the sorted data:
ax1 = plt.plot(data_sorted_p_832, p_p_832)

# 1248 x 1248
###########################################################################################

f_p = open("./pred_results/predictions_1248.txt", 'r')
cdf_p = np.array([])
line = f_p.readline()
while line:
    a = line.split()
    size = ((float(a[2])-float(a[0])) + (float(a[3])-float(a[1]))) / 2 * 0.095
    if size < size_limit:
        cdf_p = np.append(cdf_p, size)

    line = f_p.readline()

f_p.close()
print("Number of detections on 1248x1248 images: ", len(cdf_p))

data_sorted_p_1248 = sorted(cdf_p, reverse=True)
p_p_1248 = 1. * np.arange(len(cdf_p)) / (len(cdf_p) - 1) * len(cdf_p)

# plot the sorted data:
ax1 = plt.plot(data_sorted_p_1248, p_p_1248)

# 1664 x 1664
##################################################################################

f_p = open("./pred_results/predictions_1664.txt", 'r')
cdf_p = np.array([])
line = f_p.readline()
while line:
    a = line.split()
    size = ((float(a[2])-float(a[0])) + (float(a[3])-float(a[1]))) / 2 * 0.071
    if size < size_limit:
        cdf_p = np.append(cdf_p, size)

    line = f_p.readline()

f_p.close()
print("Number of detections on 1664x1664 images: ", len(cdf_p))

data_sorted_p = sorted(cdf_p, reverse=True)
p_p = 1. * np.arange(len(cdf_p)) / (len(cdf_p) - 1) * len(cdf_p)

# plot the sorted data:
ax1 = plt.plot(data_sorted_p, p_p)

# 2080 x 2080
###########################################################################################

f_p = open("./pred_results/predictions_2080.txt", 'r')
cdf_p = np.array([])
line = f_p.readline()
while line:
    a = line.split()
    size = ((float(a[2])-float(a[0])) + (float(a[3])-float(a[1]))) / 2 * 0.057
    if size < size_limit:
        cdf_p = np.append(cdf_p, size)

    line = f_p.readline()

f_p.close()
print("Number of detections on 2080x2080 images: ", len(cdf_p))

data_sorted_p_2080 = sorted(cdf_p, reverse=True)
p_p_2080 = 1. * np.arange(len(cdf_p)) / (len(cdf_p) - 1) * len(cdf_p)

# plot the sorted data:
ax1 = plt.plot(data_sorted_p_2080, p_p_2080)


###########################################################################################

ax1 = plt.legend(['Robbins dataset', 'detection_416', \
                  'detection_832', 'detection_1248', \
                  'detection_1664', 'detection_2080'], loc='upper right')
plt.show()