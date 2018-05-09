
from darkflow.net.build import TFNet
import cv2
import numpy as np

PATH_CFG_YOLO = 'cfg/yolo.cfg'
PATH_WEIGHTS_YOLO = '../weights/yolo.weights'

PATH_CFG_TINY_YOLO = 'cfg/tiny-yolo-voc.cfg'
PATH_WEIGHTS_TINY_YOLO = '../weights/tiny-yolo-voc.weights'


def predict():
    options = {"model": PATH_CFG_TINY_YOLO,
               "load": PATH_WEIGHTS_TINY_YOLO, "threshold": 0.1}

    tfnet = TFNet(options)

    imgcv = cv2.imread("./sample_img/sample_dog.jpg")
    result = tfnet.return_predict(imgcv)
    print(result)


if __name__ == '__main__':
    predict()
