"""
    This script aims to detect craters for the images in the "imgs" directory.
"""
from yolo import YOLO
from PIL import Image
import os
import os.path as osp

def listdir_nohidden(path):
    """
    Exclude hidden files in the directory.
    """
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

if __name__ == "__main__":
    yolo = YOLO()

    file_path = "./imgs"
    imlist = [osp.join(osp.realpath('.'), file_path, img) for img in listdir_nohidden(file_path)]

    # count the number of craters
    num_detection = 0
    for i in range(len(imlist)):
        image = Image.open(imlist[i])
        try:
            pred_image, top_label = yolo.detect_image(image.convert("RGB"))
        except:
            continue
        pred_image.show()
        pred_image.save("./pred_imgs/pred_%s" % (imlist[i].split('/')[-1]))
        num_detection += len(top_label)

    print("Finished!", num_detection, "craters in total.")
