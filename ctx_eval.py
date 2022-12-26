"""
    This script aims to evaluate the performance of model on CTX images, from E174°N-20° to E180°N-14°.
    The ground-truth labels were obtained from Robbins crater dataset after mapping to the CTX images.
    By running this script, Precision, recall and average IoU can be obtained.

    NOTE: The CTX tiles to be detected is in the ".CTX/cropped_images/" directory. And the ground-truth
        labels are stored in './CTX/selected_craters_XY.txt'. Both of tiles and labels can be generated
        after running 'CTX_pre-processing.py'.
"""

import numpy as np
import torch
from yolo import YOLO
from PIL import Image
import os
from utils.utils import non_max_suppression,letterbox_image,yolo_correct_boxes
import cv2
import argparse

def arg_parse():
    """
    Parse arguments to detect craters on CTX images
    """
    parser = argparse.ArgumentParser(description='evaluating on CTX images')
    parser.add_argument("--min", type=float, default=1.0, help="the size of the smallest crater to be detected")
    parser.add_argument("--max", type=float, default=32.0, help="the size of the largest crater to be detected")
    parser.add_argument("--resized_size", type=int, default=1248, help="the size of resized CTX images")

    return parser.parse_args()

def size_filter(prediction, min, max, resized_size):
    """
    This function is designed for selecting the detected craters of a certain range.

    Parameters
    ----------
    prediction: np.array, 2-d array of prediction to be filtered
    min: float or int, the lower limit of the size range
    max: float or int, the higher limit of the size range
    resized_size: int, the size of resized CTX images

    Returns
    ----------
    prediction: np.array, the filtered array with predictions in the size range
    """
    # 23710 is the size of original images, 0.005 means the resolution of original images is 5 m/pixel
    prediction = prediction[((prediction[:, 3] - prediction[:, 1]) + \
                            (prediction[:, 2] - prediction[:, 0])) / 2 > (min/(23710 * 0.005 /resized_size))]
    prediction = prediction[((prediction[:, 3] - prediction[:, 1]) + \
                            (prediction[:, 2] - prediction[:, 0])) / 2 < (max/(23710 * 0.005 /resized_size))]
    return prediction


def bbox_iou(box1, box2):
    """
    This function is referenced and modified from:
        https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/utils.py with GNU GENERAL PUBLIC LICENSE.

    This function for calculating IoU is widely used by projects relating to object detection.
    I modified it as my own version which can best fit this evaluation process.

    Parameters
    ----------
    box1, box2: 2-D tensor, box1:(Tensor[N, 4]), box2:(Tensor[M, 4]).
                Representing the a collection of bboxes with xyxy format.

    Returns
    ----------
    iou_all: np.array, 2-D numpy array with shape of (len(box1), len(box2)). iou_all[i, j] represents the IoU between
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


class ctx_Eval(YOLO):
    def ctx_evaluate(self, image):
        """
        Obtaining the array of predictions for CTX images.

        Parameters
        ----------
        image: PIL.open(image_path), the 'image_path" is the path of the ctx images.

        Returns
        ----------
        pred[:, 2:]: np.array, the array of all the predictions, but each prediction only contains the coordinates of
                    top-left and bottom-right corners of the prediction bounding box.
        """

        self.confidence = 0.5**8
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0], self.model_image_size[1])))
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
            batch_detections = non_max_suppression(output, self.config["yolov3"]["classes_num"],
                                                   conf_thres=self.confidence,
                                                   nms_thres=0.1)
        batch_detections = batch_detections[0].cpu().numpy()

        top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
        top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
        top_label = np.array(batch_detections[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detections[top_index, :4])

        top_xmin, top_ymin, top_xmax, top_ymax = \
            np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(top_bboxes[:, 1], -1), \
            np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

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

        return pred[:, 2:]


if __name__ == "__main__":
    args = arg_parse()
    ctx_eval = ctx_Eval()
    path = "./CTX/cropped_images/"
    imgs = os.listdir(path)

    prediction_all = np.array([[]])
    for img in imgs:
        if not img.startswith('.'):
            image = Image.open(path + img)
            prediction = ctx_eval.ctx_evaluate(image)
            # get the position of each tile in panoramic view from the filename of each tile
            row = int(img[12:14])
            column = int(img[15:17])
            # covert the coordinates to fit the panoramic view
            prediction[:, 0] = prediction[:, 0] + 416 * column
            prediction[:, 2] = prediction[:, 2] + 416 * column
            prediction[:, 1] = prediction[:, 1] + 416 * row
            prediction[:, 3] = prediction[:, 3] + 416 * row
            # combine all the predictions from each tile as a whole
            prediction_all = np.append(prediction_all, prediction)

    prediction_all = np.reshape(prediction_all, (-1, 4))
    # filter out the detections out of the size range
    prediction_all = size_filter(prediction_all, args.min, args.max, args.resized_size)
    np.savetxt("./CTX/predictions.txt", prediction_all, fmt='%f', delimiter=' ')

    # draw the prediction bounding boxes on the panoramic view for visualization
    image = cv2.imread('./CTX/gt_stacked_CTX.png')
    for i in range(len(prediction_all)):
        x1 = prediction_all[i, 0]
        y1 = prediction_all[i, 1]
        x2 = prediction_all[i, 2]
        y2 = prediction_all[i, 3]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.imwrite("./CTX/detections.png", image)

    # get the ground-truth labels
    robbins_path = "./CTX/selected_craters_XY.txt"
    f = open(robbins_path, 'r')
    line = f.readline()
    cnt = 0
    while line:
        a = line.split()
        if cnt == 0:
            targets = np.array([[float(a[0]), float(a[1]), float(a[2]), float(a[3])]])
        else:
            target_i = np.array([[float(a[0]), float(a[1]), float(a[2]), float(a[3])]])
            targets = np.concatenate((targets, target_i), axis=0)
        cnt += 1
        line = f.readline()

    print("Please wait. Evaluating...")
    # the IoU threshold for TP detection on the CTX images,
    # we use 0.2 and the reason is mentioned in the 5.2.1 of the report
    Iou_thres = 0.2
    target_num = len(targets)
    pred_num = len(prediction_all)
    prediction_all = torch.from_numpy(prediction_all).type(torch.LongTensor)
    targets = torch.from_numpy(targets).type(torch.LongTensor)
    iou = bbox_iou(targets, prediction_all)
    iou_valid = iou[iou > Iou_thres]
    tp_pred = len(iou_valid)

    # print the result
    print("The number of targets: ", target_num)
    print("The number of predictions: ", pred_num)
    print("The number of TP predictions: ", tp_pred)
    print("Average IoU: ", "%.2f%%" % (np.sum(iou_valid) / len(iou_valid) * 100))
    print("Precision: ", "%.2f%%" % (tp_pred / pred_num * 100))
    print("Recall: ", "%.2f%%" % (tp_pred / target_num * 100))
