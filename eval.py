
from darkflow.net.build import TFNet
import cv2
import numpy as np
import pandas as pd
from enum import Enum
from models import *
import subprocess
import os
import time
import json
import os.path


def predict(model, ckpt):

    options = {
        "model": model.get_config(),
        "load": ckpt,
        "threshold": 0.1
    }

    tfnet = TFNet(options)

    # we just count the tp since the test data only contains the labels for the one class:
    tp = 0
    count = 0
    tp_classes = {}
    count_classes = {}

    files = os.listdir(model.get_path_for_dev_images())
    for file in files:
        if "." in file:

            imgcv = cv2.imread(model.get_path_for_dev_images() + "/" + file)
            result = tfnet.return_predict(imgcv)
            print(result)

            expected_label = file.split("_")[0]

            if not expected_label in tp_classes:
                tp_classes[expected_label] = 0
                count_classes[expected_label] = 0

            count = count + 1
            count_classes[expected_label] = count_classes[expected_label] + 1

            for obj in result:
                if obj["label"] == expected_label:
                    tp = tp + 1
                    tp_classes[expected_label] = tp_classes[expected_label] + 1
                    break

    return tp, count, tp_classes, count_classes


def read_labels(file_name):
    """ Loads the labels of the imagenet synsets which should be downloaded """

    print("Reading desired labels")

    label_file = open(file_name, 'r')
    lines = label_file.readlines()
    result = []

    for line in lines:
        result.append(line.rstrip('\n'))

    label_file.close()

    return result


def get_ckpt_prefix(model):

    config_path = model.get_config()

    # it has this format: 'cfg/tiny-yolo-voc-new-1.cfg'
    return config_path.split("/")[1].split(".")[0]


def get_all_checkpoints_for_model(model):

    prefix = get_ckpt_prefix(model)
    all_ckpt_files = os.listdir("ckpt")
    result = []

    for file in all_ckpt_files:

        # files are in this format: tiny-yolo-voc-new-1-125.meta

        if prefix in file and ".meta" in file:
            number = file.split(prefix)[1].split(".meta")[0][1:]
            result.append(int(number))

    result.sort()
    return result


def get_ckpt_start(model):

    prefix = get_ckpt_prefix(model)

    try:

        label_file = open("cfg/" + prefix + ".ckpt-start", 'r')
        lines = label_file.readlines()
        label_file.close()

        return int(lines[0])

    except IOError:
        return 0

    return 0


def write_ckpt_start(model, ckpt):

    prefix = get_ckpt_prefix(model)

    text_file = open("cfg/" + prefix + ".ckpt-start", "w")
    text_file.write(str(ckpt))
    text_file.close()


def process_model(model):

    ckpts = get_all_checkpoints_for_model(model)
    labels = read_labels(model.get_labels())
    start_ckpt = get_ckpt_start(model)
    file_path = model.get_config() + ".csv"

    if os.path.isfile(file_path):
        data = pd.read_csv(file_path, sep=',',
                           encoding='utf-8', index_col=0)
    else:
        data = pd.DataFrame(data=None, index=labels, dtype=np.float64)

    for ckpt in ckpts:
        if ckpt > start_ckpt:

            tp, count, tp_classes, count_classes = predict(model, ckpt)
            series = pd.Series(dtype=np.float64)

            for label in labels:
                accuracy = tp_classes[label] / count_classes[label]
                series = series.set_value(label, accuracy)

            data[str(ckpt)] = series
            write_ckpt_start(model, ckpt)

            data.to_csv(file_path, sep=',', encoding='utf-8')
            print(data)


if __name__ == '__main__':

    process_model(Model1())  # 4750
    process_model(Model2())
    process_model(Model3())
