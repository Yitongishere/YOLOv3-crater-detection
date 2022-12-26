Config = \
{
    "yolov3": {
        # Uncomment to use the default sizes of anchor boxes of YOLOv3
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],

        # following are anchor sizes which match our selected THEMIS dataset best.
        # Obtained from"../anchors/kmeans_for_anchors.py"

        # "anchors": [[[56, 56], [84, 84], [162, 162]],
        #             [[20, 20], [28, 28], [38, 38]],
        #             [[12, 12], [14, 14], [16, 16]]],

        # as a crater detection algorithm, the classes_num should always be 1
        # when we trained the model for classifying ouverlapping craters, set it as 2
        "classes_num": 1,
    },
    "img_h": 416,
    "img_w": 416,
}
