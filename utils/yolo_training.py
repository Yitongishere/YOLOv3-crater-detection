"""
    This script includes the functions that used for training the model.
    Ideals and code are referenced and modified from:
    https://github.com/bubbliiiing/yolo3-pytorch/blob/master/nets/yolo_training.py with MIT LICENSE
"""

from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
from utils.utils import bbox_iou


def jaccard(_box_a, _box_b):
    """
    This function for calculating IoU of box_a and box_b.

    Parameters
    ----------
    _box_a, _box_b: np.array, box1:([N, 4]), box2:(Tensor[M, 4]).
                Representing the a collection of bboxes with xywh format.

    Returns
    ----------
    inter / union: 2-D numpy array, with shape of (len(_box_a), len(_box_b)).
    """

    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]
    # calculating the area of anchor box and ground-truth bbox
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
             (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # iou
    union = area_a + area_b - inter
    return inter / union  # [A,B]

    
def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSELoss(pred,target):
    return (pred-target)**2

def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, cuda):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.feature_length = [img_size[0]//32,img_size[0]//16,img_size[0]//8]
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.cuda = cuda

    def forward(self, input, targets=None):
        # get the batch size
        bs = input.size(0)

        in_h = input.size(2)
        in_w = input.size(3)

        # calculating the stride
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        # scaled_anchors for fitting input image
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        
        # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
        prediction = input.view(bs, int(self.num_anchors/3),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        # decoding the predictions
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # get the final predictions and compare with the ground-truth labels
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y =\
                                                                            self.get_target(targets, scaled_anchors,
                                                                                            in_w, in_h)

        noobj_mask, all_ious = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)
        if self.cuda:
            box_loss_scale_x = (box_loss_scale_x).cuda()
            box_loss_scale_y = (box_loss_scale_y).cuda()
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
        box_loss_scale = 2 - box_loss_scale_x*box_loss_scale_y
        
        # losses.
        loss_x = torch.sum(BCELoss(x, tx) / bs * box_loss_scale * mask)
        loss_y = torch.sum(BCELoss(y, ty) / bs * box_loss_scale * mask)
        loss_w = torch.sum(MSELoss(w, tw) / bs * 0.5 * box_loss_scale * mask)
        loss_h = torch.sum(MSELoss(h, th) / bs * 0.5 * box_loss_scale * mask)
        loss_conf = torch.sum(BCELoss(conf, mask) * mask / bs) + torch.sum(BCELoss(conf, mask) * noobj_mask / bs)
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1])/bs)
        # sum the loss of each part
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
                loss_h.item(), loss_conf.item(), loss_cls.item(), all_ious

    def get_target(self, target, anchors, in_w, in_h):
        """
        This function is for converting the ground-truth labels to the format of yolov3 outputs

        Parameters
        ----------
        target: ground-truth labels
        anchors: anchor boxes
        in_w: width of input image
        in_h: height of input image

        Returns
        ----------
        the ground-truth labels with the format of yolov3 outputs
        """
        bs = len(target)
        # get the anchors
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][self.feature_length.index(in_w)]
        subtract_index = [0,3,6][self.feature_length.index(in_w)]
        # create the tensors for representing there is or not an object for that position
        mask = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)

        # "t" means target
        tx = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)
        for b in range(bs):
            for t in range(target[b].shape[0]):
                # calculate the position in terms of the feature map
                gx = target[b][t, 0] * in_w
                gy = target[b][t, 1] * in_h
                
                gw = target[b][t, 2] * in_w
                gh = target[b][t, 3] * in_h

                # calculate the position of the grid that groud-truth target belongs to
                gi = int(gx)
                gj = int(gy)

                # calculate the actual position of ground-truth bbox
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                
                # calculate the position of each anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # calculate the iou between anchor boxes and ground-truth box
                anch_ious = bbox_iou(gt_box, anchor_shapes)
               
                # Find the best matching anchor box for each ground-truth box
                best_n = np.argmax(anch_ious)
                if best_n not in anchor_index:
                    continue
                # Masks
                if (gj < in_h) and (gi < in_w):
                    best_n = best_n - subtract_index
                    # finding the bboxes which include a object
                    noobj_mask[b, best_n, gj, gi] = 0
                    mask[b, best_n, gj, gi] = 1
                    # encoding the groud-truth labels
                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n+subtract_index][0])
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n+subtract_index][1])
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][t, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][t, 3]
                    # setting confidence for ground-truth as 1
                    tconf[b, best_n, gj, gi] = 1
                    # class
                    tcls[b, best_n, gj, gi, int(target[b][t, 4])] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, noobj_mask):
        all_ious = []
        bs = len(target)
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]

        # decoding predictions
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # generating the grid. the center of anchor box is the top-left corner of the grid
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
            int(bs*self.num_anchors/3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
            int(bs*self.num_anchors/3), 1, 1).view(y.shape).type(FloatTensor)

        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        
        # decoding predictions
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)

            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(np.concatenate([gx, gy, gw, gh],-1)).type(FloatTensor)

                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
                for t in range(target[i].shape[0]):
                    anch_iou = anch_ious[t].view(pred_boxes[i].size()[:3])
                    noobj_mask[i][anch_iou>self.ignore_threshold] = 0
                    all_ious.append(anch_iou)
        return noobj_mask, all_ious


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Generator(object):
    def __init__(self,batch_size,train_lines, image_size,):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size

    # data augmentation
    def get_random_data(self, annotation_line, input_shape, jitter=.1, hue=.1, sat=1.3, val=1.3):
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x)*255 # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:,:4]>0).any():
            return image_data, box_data
        else:
            return image_data, []

    # load data
    def generate(self):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            inputs = []
            targets = []
            for annotation_line in lines:  
                img, y = self.get_random_data(annotation_line, self.image_size[0:2])

                if len(y)!=0:
                    boxes = np.array(y[:,:4],dtype=np.float32)
                    boxes[:,0] = boxes[:,0]/self.image_size[1]
                    boxes[:,1] = boxes[:,1]/self.image_size[0]
                    boxes[:,2] = boxes[:,2]/self.image_size[1]
                    boxes[:,3] = boxes[:,3]/self.image_size[0]

                    # convert xyxy to xywh
                    boxes = np.maximum(np.minimum(boxes,1),0)
                    boxes[:,2] = boxes[:,2] - boxes[:,0]
                    boxes[:,3] = boxes[:,3] - boxes[:,1]
    
                    boxes[:,0] = boxes[:,0] + boxes[:,2]/2
                    boxes[:,1] = boxes[:,1] + boxes[:,3]/2
                    y = np.concatenate([boxes,y[:,-1:]],axis=-1)
                img = np.array(img,dtype = np.float32)

                inputs.append(np.transpose(img/255.0,(2,0,1)))                  
                targets.append(np.array(y,dtype = np.float32))
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets



