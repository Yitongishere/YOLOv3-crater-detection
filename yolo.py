"""
    This script includes the class definition of YOLOv3 model.
    Ideals and code are referenced and modified from:
    https://github.com/qqwweee/keras-yolo3/blob/master/yolo.py with MIT LICENSE
    and https://github.com/bubbliiiing/yolo3-pytorch/blob/master/yolo.py with MIT LICENSE
"""

import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from network.yolov3_architecture import YoloLayer
from PIL import ImageFont, ImageDraw
from cfg.config import Config
from utils.utils import non_max_suppression, DecodeBox, letterbox_image, yolo_correct_boxes


class YOLO(object):

    # set the weights file to be loaded, txt file contains the class name, confidence threshold
    _defaults = {
        "model_path": 'weights/Epoch50-Total_Loss6.2297-Val_Loss6.6023.pth',
        "classes_path": 'cfg/crater_classes.txt',
        "model_image_size": (416, 416, 3),
        "confidence": 0.2,
        "cuda": False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # initialize yolov3
    def __init__(self):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.config = Config
        self.generate()

    # get the class name
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def generate(self):
        self.config["yolov3"]["classes_num"] = len(self.class_names)
        self.net = YoloLayer(self.config)

        # using cuda if there is a GPU available
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        # convert the output of yolov3
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.config["yolov3"]["anchors"][i], self.config["yolov3"]["classes_num"],  (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # setting the colors of bounding boxes for visualization
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))



    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))
        photo = np.array(crop_img,dtype = np.float32)
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
        try :
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            # if there is no bbox left after nms, then return the original image
            return image

        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
        font = ImageFont.truetype(font='cfg/cmb10.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            # make bounding boxes little bit larger when visualizing on original images to avoid blocking objects
            top = top - 2
            left = left - 2
            bottom = bottom + 2
            right = right + 2
            # convert to int
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # drawing bounding boxes on original images
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
        # Show tags or hide tags
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image, top_label

