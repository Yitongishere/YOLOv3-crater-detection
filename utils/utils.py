"""
    This script includes the functions that used for making final predictions after outputs of yolov3.
    Ideals and code are referenced and modified from:
    https://github.com/bubbliiiing/yolo3-pytorch/tree/master/utils with MIT LICENSE
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

def bbox_iou(box1, box2):
    """
    Calculating IoUs of boxes.

    Parameters
    ----------
    box1, box2: 2-D tensor, box1:(Tensor[N, 4]), box2:(Tensor[M, 4]).
                Representing the a collection of bboxes with xyxy format.

    Returns
    ----------
    iou: Tensor[N, M], the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # Intersection area
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.2, nms_thres=0.3):
    """
    Non-Maximum Suppression for filtering the predictions with the confidence lower than the conf_thres. And also
    in the cases when multiple bounding boxes with IoU higher than the nms_thres, only the bounding box with the
    highest confidence will be remain.

    Parameters
    ----------
    prediction: np.array, all of the prediction bounding boxes before filtering
    num_classes: int, the number of class
    conf_thres: float, confidence threshold
    nms_thres: float, non-maximum supression threshold

    Returns
    ----------
    output: np.array, the final prediction after filtering
    """
    # convert the format of labels
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # first filtering by using conf_thres
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue

        # get class_conf and class name
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # get class
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # the predictions after the first filtering
            remaining_detection = detections[detections[:, -1] == c]
            # sort by confidence from high to low
            _, conf_sort_index = torch.sort(remaining_detection[:, 4], descending=True)
            remaining_detection = remaining_detection[conf_sort_index]
            # Non-Maximum Suppression
            save_zone = []   # play a role as the "save zone" to store the final predictions
            while remaining_detection.size(0):
                # pick out the prediction with the highest conf
                save_zone.append(remaining_detection[0].unsqueeze(0))
                # if no predictions remain, end the loop
                if len(remaining_detection) == 1:
                    break
                # calculate IoUs between the highest conf one and each of remaining predictions
                ious = bbox_iou(save_zone[-1], remaining_detection[1:])
                # check if any remaining predictions with IoU larger than the nms_thres. If yes, cut them off
                remaining_detection = remaining_detection[1:][ious < nms_thres]
            # list all of the final prediction in the "save zone"
            save_zone = torch.cat(save_zone).data
            # Add max detections to outputs
            output[image_i] = save_zone if output[image_i] is None else torch.cat((output[image_i], save_zone))

    return output


def letterbox_image(image, size):
    """
    Resize an arbitrarily sized image to 416 by 416 pixels as input size.
    Resize the longer side to 416 pixel and keep the aspect ratio intact
    for shorter side. Padding the left-out portions as black.

    Parameters
    ----------
    image: the input image
    size: the image size after resizing, 416 for yolov3

    Returns
    ----------
    new_image: resized image
    """
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    """
    Match the prediction bounding boxes on the actual input image.

    Parameters
    ----------
    top: float, top-left-y
    left: float, top-left-x
    bottom: float, bottom-right-y
    right: float, bottom-right-x

    Returns
    ----------
    boxes: np.array, the prediction bounding boxes that match with actual input image
    """
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[:, 0:1], box_mins[:, 1:2], box_maxes[:, 0:1], box_maxes[:, 1:2]],axis=-1)
    print(np.shape(boxes))
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        # stride for different feature maps
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width
        # normalize the anchor size for each feature map
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

        # resize the prediction
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # extract the original outputs of yolov3 neural network for later decoding
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # generating the grids on images, with top-left corner of each gird as the center of the anchor box
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # generating the width and height of the anchor boxes
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # Decoding. get the normalized center coordinates (x and y), width, height in terms of the panoramic image
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # scale the normalized prediction to 416 by 416 image
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data






