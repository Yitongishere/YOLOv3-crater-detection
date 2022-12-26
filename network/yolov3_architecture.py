"""
    This script is the construction of the YOLOv3 with backbone of DarkNet53
    Ideals and code are referenced and modified from:
    https://github.com/BobLiu20/YOLOv3_PyTorch/blob/master/nets/backbone/darknet.py
    and https://github.com/bubbliiiing/yolo3-pytorch/blob/master/nets/yolo3.py with MIT LICENSE
"""

import torch
import torch.nn as nn
import math
from collections import OrderedDict


###################################################################################
# Backbone: DarkNet 53

# Residual Block used repeatedly in the YOLOv3, containing 2 Conv2Ds and each of them followed by BN and LeakyReLU
# As the Residual Block is used repetitively in the architecture, so it was packed for convenient
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(ResidualBlock, self).__init__()
        # 1x1 conv layer
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu1 = nn.LeakyReLU(0.1)
        # 3x3 conv layer
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # residual layer
        x += residual
        return x

# The Downsampling Block before the Residual Blocks
class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(DownsamplingBlock, self).__init__()
        self.ds_conv = nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.ds_bn = nn.BatchNorm2d(channels)
        self.ds_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.ds_conv(x)
        x = self.ds_bn(x)
        x = self.ds_relu(x)
        return x

# It is the general structure of the DarkNet
# For DarkNet with different layers, just need to use different parameter for repetitive times of Residual Blocks
class DarkNet(nn.Module):
    # repeat_times: a list contains the repeat times of Residual block
    def __init__(self, repeat_times):
        super(DarkNet, self).__init__()
        self.in_channels = 32
        # the Conv layer before the first Residual block
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu1 = nn.LeakyReLU(0.1)

        # 5 down-sampling + residual block with different repeat time
        self.layer1 = self.concat_layers([32, 64], repeat_times[0])
        self.layer2 = self.concat_layers([64, 128], repeat_times[1])
        self.layer3 = self.concat_layers([128, 256], repeat_times[2])
        self.layer4 = self.concat_layers([256, 512], repeat_times[3])
        self.layer5 = self.concat_layers([512, 1024], repeat_times[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # initializing weights for conv and bn layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def concat_layers(self, channels, blocks):
        ds_and_res = []
        # Down-sampling, with stride as 2, kernel size as 3 by 3.
        ds_and_res.append(("ds_conv", nn.Conv2d(self.in_channels, channels[1], kernel_size=3,stride=2, padding=1, bias=False)))
        ds_and_res.append(("ds_bn", nn.BatchNorm2d(channels[1])))
        ds_and_res.append(("ds_relu", nn.LeakyReLU(0.1)))
        # append Residual blocks for required times after Down-sampling
        self.in_channels = channels[1]
        for i in range(0, blocks):
            ds_and_res.append(("residual_{}".format(i), ResidualBlock(self.in_channels, channels)))
        # string the Down-sampling layers and Residual Blocks
        ds_and_res_layer = nn.Sequential(OrderedDict(ds_and_res))
        return ds_and_res_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # save the 52x52, 26x26 and 13x13 feature maps for later concat
        output_52 = self.layer3(x)
        output_26 = self.layer4(output_52)
        output_13 = self.layer5(output_26)
        return output_52, output_26, output_13

# Get DarkNet53 by setting the repeat times as [1,2,8,8,4]
def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model


###################################################################################
# YOLO Layer

# Pack conv2d, bn and relu for concise
def convolutional_layer(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    packed_conv_layer = nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))
    return packed_conv_layer

# Pack the convolutional block before the final output of YOLOv3
def convolutional_block(filters_list, in_filters, out_filter):
    conv_block = nn.ModuleList([
        convolutional_layer(in_filters, filters_list[0], 1),
        convolutional_layer(filters_list[0], filters_list[1], 3),
        convolutional_layer(filters_list[1], filters_list[0], 1),
        convolutional_layer(filters_list[0], filters_list[1], 3),
        convolutional_layer(filters_list[1], filters_list[0], 1),
        convolutional_layer(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    ])
    return conv_block

# Combine the backbone and yolo layer together to finish the whole yolov3
class YoloLayer(nn.Module):
    def __init__(self, config):
        super(YoloLayer, self).__init__()
        # the config file for easier setting anchor sizes, number of class, input image size
        self.config = config
        # backbone as DarkNet53
        self.backbone = darknet53()

        out_filters = self.backbone.layers_out_filters
        # branch for 13x13 feature map
        final_out_filter0 = len(config["yolov3"]["anchors"][0]) * (5 + config["yolov3"]["classes_num"])
        self.last_layer0 = convolutional_block([512, 1024], out_filters[-1], final_out_filter0)

        # branch for 26x26 feature map
        final_out_filter1 = len(config["yolov3"]["anchors"][1]) * (5 + config["yolov3"]["classes_num"])
        self.last_layer1_conv = convolutional_layer(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = convolutional_block([256, 512], out_filters[-2] + 256, final_out_filter1)

        # branch for 52x52 feature map
        final_out_filter2 = len(config["yolov3"]["anchors"][2]) * (5 + config["yolov3"]["classes_num"])
        self.last_layer2_conv = convolutional_layer(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = convolutional_block([128, 256], out_filters[-3] + 128, final_out_filter2)


    def forward(self, x):

        # function for saving the branch to be concatenated and outputting branch result
        def branch(last_layer, in_branch):
            for i, e in enumerate(last_layer):
                in_branch = e(in_branch)
                if i == 4:
                    out_branch = in_branch
            return in_branch, out_branch

        # import backbone
        x_52, x_26, x_13 = self.backbone(x)
        #  branch for 13x13
        final_13, branch_13 = branch(self.last_layer0, x_13)
        # branch for 26x26
        branch_26 = self.last_layer1_conv(branch_13)
        branch_26 = self.last_layer1_upsample(branch_26)
        branch_26 = torch.cat([branch_26, x_26], 1)
        final_26, branch_26 = branch(self.last_layer1, branch_26)
        # branch for 52x52
        branch_52 = self.last_layer2_conv(branch_26)
        branch_52 = self.last_layer2_upsample(branch_52)
        branch_52 = torch.cat([branch_52, x_52], 1)
        final_52, branch_52 = branch(self.last_layer2, branch_52)

        return final_13, final_26, final_52
