"""
    This script aims to get the list of trainval samples and the list of test samples
    which server for the evaluation progress. From the list we can extract the corresponding
    images and their ground-truth targets and after comparing with the prediction of the model,
    we can get the average IoU, precision, recall to evaluate our performance.
"""

import os

anno_trainval = r'./Annotations_trainval'
anno_test = r'./Annotations_test'
save_path = r'./'

a_trainval = os.listdir(anno_trainval)
total_trainval = []
a_test = os.listdir(anno_test)
total_test = []

for trainval in a_trainval:
    if trainval.endswith(".txt"):
        total_trainval.append(trainval)
for test in a_test:
    if test.endswith(".txt"):
        total_test.append(test)
print("TrainVal list: ", total_trainval)
print("Test list: ", total_test)

num_trainval = len(total_trainval)
ftrainval = open(os.path.join(save_path,'trainval.txt'), 'w')
num_test = len(total_test)
ftest = open(os.path.join(save_path,'test.txt'), 'w')

for i in range(num_trainval):
    if i != num_trainval - 1:
        name = total_trainval[i][:-4]+'\n'
    else:
        name = total_trainval[i][:-4]
    ftrainval.write(name)
for i in range(num_test):
    if i != num_test - 1:
        name = total_test[i][:-4]+'\n'
    else:
        name = total_test[i][:-4]
    ftest.write(name)

ftrainval.close()
ftest.close()
