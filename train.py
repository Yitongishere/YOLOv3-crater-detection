"""
    This script is for train the yolov3 neural network with our THEMIS dataset
    Ideals and code are referenced and modified from:
    https://github.com/bubbliiiing/yolo3-pytorch/blob/master/train.py with MIT LICENSE
"""

import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from cfg.config import Config
from network.yolov3_architecture import YoloLayer
from utils.yolo_training import YOLOLoss, Generator
import matplotlib.pyplot as plt
import math
import argparse
import warnings
warnings.filterwarnings('ignore')

def arg_parse():
    """
    Parse arguments to train the YOLOv3
    """
    parser = argparse.ArgumentParser(description='YOLOv3 training')
    parser.add_argument("--CUDA", type=bool, default=False, help="use CUDA or not")
    parser.add_argument("--weights", type=str, default="./weights/yolo_weights.pth", help="pre-trained to be loaded")
    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
    parser.add_argument("--end_epoch", type=int, default=70, help="end epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--bs", type=int, default=8, help="batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="proportion of validation set")

    return parser.parse_args()


def fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    """
    Train the model for one epoch.
    """
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_size:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
        # settings
        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        iou = []

        # for 3 feature maps
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
            iou.append(loss_item[7])

        # getting the iou
        iou_collect = np.array([])
        for j in range(3):
            for k in range(3):
                try:
                    iou_ = iou[j][k].cpu().numpy()
                    iou_collect = np.append(iou_collect, iou_)
                    iou_collect = np.sort(iou_collect)[-5:]
                except IndexError:
                    continue


        if len(iou_collect > 0):
            average_iou = np.sum(iou_collect)/len(iou_collect)
        else:
            average_iou = 0

        loss = sum(losses)
        loss.backward()
        optimizer.step()

        # sum up the loss
        total_loss += loss
        waste_time = time.time() - start_time
        print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('iter:' + str(iteration+1) + '/' + str(epoch_size) + ' || Total Loss: %.4f || Average IoU: %.4f || %.4fs/step' % (total_loss/(iteration+1), average_iou, waste_time))
        start_time = time.time()

    print('Start Validation')
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_size_val:
            break
        images_val, targets_val = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            else:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            iou_val = []

            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item[0])
                iou_val.append(loss_item[7])

            iou_collect_val = np.array([])
            for j in range(3):
                for k in range(3):
                    try:
                        iou_v = iou_val[j][k].cpu().numpy()
                        iou_collect_val = np.append(iou_collect_val, iou_v)
                        iou_collect_val = np.sort(iou_collect)[-5:]
                    except IndexError:
                        continue

            if len(iou_collect_val > 0):
                average_iou_val = np.sum(iou_collect_val) / len(iou_collect_val)
            else:
                average_iou_val = 0

            loss = sum(losses)
            val_loss += loss

    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size),val_loss/(epoch_size_val)))
    print('Average IoU: %.4f || Val Average IoU: %.4f ' % (average_iou, average_iou_val))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), './weights/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size),val_loss/(epoch_size_val)))

    return total_loss/epoch_size, val_loss/epoch_size_val, average_iou, average_iou_val


if __name__ == "__main__":
    args = arg_parse()

    # initializing
    annotation_path = './Data/data_train.txt'
    model = YoloLayer(Config)
    Cuda = args.CUDA

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.weights, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('Training...')

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # Loss function
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolov3"]["anchors"],[-1,2]),
                                    Config["yolov3"]["classes_num"], (Config["img_w"], Config["img_h"]), Cuda))

    # split validation set and training set
    val_split = args.val_split
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(42)
    np.random.shuffle(lines)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # record for monitoring training process
    train_loss = []
    validation_loss = []
    train_Ave_iou = []
    validation_Ave_iou = []
    ep = []

    if True:
        # hyperparameters setting
        lr = args.lr
        Batch_size = args.bs
        Start_Epoch = args.start_epoch
        End_Epoch = args.end_epoch

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95, last_epoch=-1)

        # load data
        gen = Generator(Batch_size, lines[:num_train],
                         (Config["img_h"], Config["img_w"])).generate()
        gen_val = Generator(Batch_size, lines[num_train:],
                         (Config["img_h"], Config["img_w"])).generate()

        epoch_size = math.ceil(num_train/Batch_size)
        epoch_size_val = math.ceil(num_val/Batch_size)


        for param in model.backbone.parameters():
            param.requires_grad = True

        # train
        for epoch in range(Start_Epoch, End_Epoch):
            tl, vl, ti, vi = fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,End_Epoch,Cuda)
            lr_scheduler.step()
            train_loss.append(tl)
            validation_loss.append(vl)
            train_Ave_iou.append(ti)
            validation_Ave_iou.append(vi)
            ep.append(Start_Epoch + epoch + 1)

    # monitoring the training process by loss and average iou
    ax1 = plt.plot(ep, train_loss, label='train loss')
    ax1 = plt.plot(ep, validation_loss, label='validation loss')
    ax1 = plt.title('Model Loss')
    ax1 = plt.ylim(0, 50)
    ax1 = plt.ylabel('Loss')
    ax1 = plt.xlabel('Epoch')
    ax1 = plt.legend(['train', 'validation'], loc='upper left')
    ax1 = plt.savefig('./Loss.png')
    ax1 = plt.show()

    ax2 = plt.plot(ep, train_Ave_iou, label='train Ave IoU')
    ax2 = plt.plot(ep, validation_Ave_iou, label='validation Ave IoU')
    ax2 = plt.title('Model Ave IoU')
    ax1 = plt.ylim(0, 1)
    ax2 = plt.ylabel('Ave IoU')
    ax2 = plt.xlabel('Epoch')
    ax2 = plt.legend(['train', 'validation'], loc='upper left')
    ax2 = plt.savefig('./Ave_iou.png')
    ax2 = plt.show()









