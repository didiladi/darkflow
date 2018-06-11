
from darkflow.net.build import TFNet
import cv2
import numpy as np
import pandas as pd
from enum import Enum
from models import Model
import subprocess
import os
import time
import json
import os.path


def predict(model, ckpt):

    options = {
        "model": model.get_config(),
        "labels": model.get_labels(),
        "load": ckpt,
        "threshold": 0.1
    }

    tfnet = TFNet(options)

    # we just count the tp since the test data only contains the labels for the one class:
    tp = 0
    count = 0
    tp_classes = {}
    count_classes = {}

    files = os.listdir(model.get_path_for_test_images())
    for file in files:
        if "." in file:

            try:

                imgcv = cv2.imread(
                    model.get_path_for_test_images() + "/" + file)

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

            except AssertionError:
                continue

    return tp, count, tp_classes, count_classes


def get_all_checkpoints_for_model(model):

    prefix = model.get_ckpt_prefix()
    all_ckpt_files = os.listdir("ckpt")
    result = []

    for file in all_ckpt_files:

        # files are in this format: tiny-yolo-voc-new-1-125.meta

        if prefix in file and ".meta" in file:
            number = file.split(prefix)[1].split(".meta")[0][1:]
            result.append(int(number))

    result.sort()
    return result


def process_model(model):

    data = model.get_evaluation_data()
    ckpts = get_all_checkpoints_for_model(model)
    labels = model.read_labels()
    start_ckpt = model.get_ckpt_start()

    for ckpt in ckpts:
        if ckpt > start_ckpt:

            tp, count, tp_classes, count_classes = predict(model, ckpt)

            series = pd.Series(dtype=np.float64)

            for label in labels:
                accuracy = tp_classes[label] / count_classes[label]
                series = series.set_value(label, accuracy)

            data[str(ckpt)] = series
            model.write_ckpt_start(ckpt)

            model.save_evaluation_data(data)
            print(data)


if __name__ == '__main__':

    for i in range(0, 9):
        process_model(Model(i+1))
