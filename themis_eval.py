"""
    This script aims to evaluate the performance of model on the 356 THEMIS images, which were not included
    in the training process. By running this script, Precision, recall and average IoU can be obtained.
"""

import numpy as np
import time
import torch
from yolo import YOLO
from PIL import Image
import argparse
from utils.utils import non_max_suppression, letterbox_image, yolo_correct_boxes



def arg_parse():
    """
    Parse arguments to detect craters on THEMIS images
    """
    parser = argparse.ArgumentParser(description='evaluating on THEMIS images')
    parser.add_argument("--iou_thres", type=float, default=0.3, help="the IoU threshold for true positive detection")

    return parser.parse_args()



def bbox_iou(box1, box2):
    """
    This function is referenced and modified from:
        https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/utils.py with GNU GENERAL PUBLIC LICENSE.

    This function for calculating IoU is widely used by projects relating to object detection.
    I modified it as my own version which can best fit our THEMIS evaluation process.

    Parameters
    ----------
    box1, box2: 2-D tensor, box1:(Tensor[N, 4]), box2:(Tensor[M, 4]).
                Representing the a collection of bboxes with xyxy format.

    Returns
    ----------
    iou_all: 2-D numpy array, with shape of (len(box1), len(box2)). iou_all[i, j] represents the IoU between
            box1[i] and box2[j].
    """

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    for i in range(len(box1)):
        inter_rect_x1 = torch.max(b1_x1[i], b2_x1)
        inter_rect_y1 = torch.max(b1_y1[i], b2_y1)
        inter_rect_x2 = torch.min(b1_x2[i], b2_x2)
        inter_rect_y2 = torch.min(b1_y2[i], b2_y2)

        for j in range(len(box2)):
            inter_area = torch.clamp(inter_rect_x2[j] - inter_rect_x1[j] + 1, min=0) * \
                        torch.clamp(inter_rect_y2[j] - inter_rect_y1[j] + 1, min=0)

            b1_area = (b1_x2[i] - b1_x1[i] + 1) * (b1_y2[i] - b1_y1[i] + 1)
            b2_area = (b2_x2[j] - b2_x1[j] + 1) * (b2_y2[j] - b2_y1[j] + 1)

            iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

            if j == 0:
                iou_i = np.array([[iou.item()]])
            else:
                iou_i = np.append(iou_i, iou.item())

        if i == 0:
            iou_all = np.array([iou_i])
        else:
            iou_all = np.concatenate((iou_all, [iou_i]), axis=0)

    return iou_all


class Eval(YOLO):
    def evaluate(self, image_id, image):
        """
        Obtaining the array of ground-truth targets and the array of predictions.

        Parameters
        ----------
        image_id: string, the file name of the image to be detected
        image: PIL.open(image_path), the 'image_path" is the path of the 'image_id'

        Returns
        ----------
        pred: np.array, the 2-d numpy array with shape of (the number of detections, 6). The 6 entries represent
            class name, confidence, topleft-x, topleft-y, bottomright-x and bottomright-y.
        target: np.array, the 2-d numpy array read from the annotations, (the number of ground-truth craters, 5)
            The 5 entries represent ground-truth class name, top-left-x, top-left-y, bottom-right-x and bottom-right-y.
        """

        # set confidence as 0.2 (consistent with Benedix work)
        self.confidence = 0.2

        image_shape = np.array(np.shape(image)[0:2])
        crop_img = np.array(letterbox_image(image, (416, 416)))
        photo = np.array(crop_img, dtype=np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)

        images = np.asarray(images)
        images = torch.from_numpy(images)
        if self.cuda:
            images = images.cuda()

        with torch.no_grad():
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            final_detections = non_max_suppression(output, self.config["yolov3"]["classes_num"],
                                                   conf_thres=self.confidence,
                                                   nms_thres=0.1)
        # if no bbox left after nms, return the original image
        try:
            final_detections = final_detections[0].cpu().numpy()
        except:
            return image

        top_index = final_detections[:, 4] * final_detections[:, 5] > self.confidence
        top_conf = final_detections[top_index, 4] * final_detections[top_index, 5]
        top_label = np.array(final_detections[top_index, -1], np.int32)
        top_bboxes = np.array(final_detections[top_index, :4])

        top_xmin, top_ymin, top_xmax, top_ymax = \
            np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1], -1), \
            np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

        # the bounding boxes of predictions
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = float(top_conf[i])
            score = float('%.4f' % score)

            top, left, bottom, right = boxes[i]
            if i == 0:
                pred = np.array([[int(predicted_class), score, int(left), int(top), int(right), int(bottom)]])
            else:
                pred_i = np.array([[int(predicted_class), score, int(left), int(top), int(right), int(bottom)]])
                pred = np.concatenate((pred, pred_i), axis=0)

        # read the ground-truth targets from the annotations
        image_path = "./Data/Annotations_test/" + image_id + ".txt"
        f = open(image_path, 'r')
        line = f.readline()
        cnt = 0
        while line:
            a = line.split()
            if cnt == 0:
                target = np.array([[int(a[0]),
                                    int(float(a[1]) * 416 - 0.5 * float(a[3]) * 416),
                                    int(float(a[2]) * 416 - 0.5 * float(a[4]) * 416),
                                    int(float(a[1]) * 416 + 0.5 * float(a[3]) * 416),
                                    int(float(a[2]) * 416 + 0.5 * float(a[4]) * 416)]])
            else:
                target_i = np.array([[int(a[0]),
                                    int(float(a[1]) * 416 - 0.5 * float(a[3]) * 416),
                                    int(float(a[2]) * 416 - 0.5 * float(a[4]) * 416),
                                    int(float(a[1]) * 416 + 0.5 * float(a[3]) * 416),
                                    int(float(a[2]) * 416 + 0.5 * float(a[4]) * 416)]])
                target = np.concatenate((target, target_i), axis=0)
            cnt += 1
            line = f.readline()

        return pred, target


if __name__ == "__main__":
    args = arg_parse()

    # the IoU threshold for true positive detections, use 0.5 here for being consistent with Benedix work
    Iou_thres = args.iou_thres
    eval = Eval()
    # get the file names of specific annotations by reading the list of test set
    image_ids = open('./Data/test.txt').read().strip().split()
    iou_all = np.array([])
    # counting the number of targets, predictions
    target_num = 0
    pred_num = 0
    # number of images to be tested, 356 for our test set
    test_num = len(image_ids)
    cnt = 0
    start_time = time.time()
    for image_id in image_ids:
        cnt += 1
        image_path = "./Data/data_images/" + image_id + ".png"
        image = Image.open(image_path)
        try:
            pred, target = eval.evaluate(image_id, image)
            pred_bbox = torch.from_numpy(pred[:,2:]).type(torch.LongTensor)
            target_bbox = torch.from_numpy(target[:,1:]).type(torch.LongTensor)

            # compare the array of targets and the array of prediction
            iou = bbox_iou(target_bbox, pred_bbox)
            # for each target, only remain the prediction with the highest IoU
            iou = np.amax(iou, axis=1)
            # also filter out the predictions with IoU lower than IoU threshold
            iou = iou[iou > Iou_thres]

            # combine all the predictions from each image as a whole
            iou_all = np.append(iou_all, iou)
            target_num += len(target)
            pred_num += len(pred)
        except: # if no crater has been detected
            target_num += len(open("./Data/Annotations_test/" + image_id + ".txt", 'r').readlines())
            print(cnt, "/", test_num, " finished!")
            continue
        print(cnt,"/",test_num," finished!")

    # calculating average IoU, precision, recall
    waste_time = time.time() - start_time
    print("waste time: ", waste_time)
    average_iou = np.sum(iou_all) / len(iou_all)
    true_pred_num = len(iou_all)
    precision = true_pred_num / pred_num
    recall = true_pred_num / target_num

    # print the result
    print("The number of targets: ", target_num)
    print("The number of predictions: ", pred_num)
    print("The number of TP predictions: ", true_pred_num)
    print("Average IoU: ", "%.2f%%" % (average_iou * 100))
    print("Precision: ", "%.2f%%" % (precision * 100))
    print("Recall: ", "%.2f%%" % (recall * 100))

