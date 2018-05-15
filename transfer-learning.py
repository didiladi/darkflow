
from darkflow.net.build import TFNet
import cv2
import numpy as np
from enum import Enum
from models import *
import subprocess

PATH_CFG_YOLO = 'cfg/yolo.cfg'
PATH_WEIGHTS_YOLO = '../weights/yolo.weights'

PATH_CFG_TINY_YOLO = 'cfg/tiny-yolo-voc.cfg'
PATH_WEIGHTS_TINY_YOLO = '../weights/tiny-yolo-voc.weights'


def train_model(model, load_ckpt=False):
    ''' Trains the given model '''

    cmd = [
        './flow',
        '--model', model.get_config(),
        '--train',
        '--dataset', model.get_path_for_train_images(),
        '--annotation', model.get_path_for_train_annotations(),
        '--load'
    ]
    if load_ckpt:
        cmd.append('-1')
    else:
        cmd.append(model.get_weights())

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    process.wait()
    for line in process.stderr:
        print(line)


def predict():

    options = {
        "model": PATH_CFG_TINY_YOLO,
        "load": PATH_WEIGHTS_TINY_YOLO,
        "threshold": 0.1
    }

    tfnet = TFNet(options)

    imgcv = cv2.imread("./sample_img/sample_dog.jpg")
    result = tfnet.return_predict(imgcv)
    tfnet.train
    print(result)


if __name__ == '__main__':
    train_model(Model1())
