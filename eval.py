
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
    tp_classes = {}
    fp_classes = {}
    fn_classes = {}
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
                    fp_classes[expected_label] = 0
                    fn_classes[expected_label] = 0
                    count_classes[expected_label] = 0

                count_classes[expected_label] = count_classes[expected_label] + 1

                num_distinct_classes = 0
                already_handled = []
                correct_label_available = False

                for obj in result:

                    if obj["label"] == expected_label:
                        correct_label_available

                    if obj["label"] not in already_handled:
                        already_handled.append(obj["label"])
                        num_distinct_classes = num_distinct_classes + 1

                already_handled = []

                for obj in result:

                    if obj["label"] not in already_handled:
                        already_handled.append(obj["label"])

                        if obj["label"] == expected_label:
                            tp_classes[expected_label] = tp_classes[expected_label] + \
                                1 / num_distinct_classes
                        else:

                            minus = 0
                            if correct_label_available:
                                minus = 1

                            fp_classes[expected_label] = fp_classes[expected_label] + \
                                (num_distinct_classes - minus) / \
                                num_distinct_classes

                if len(result) == 0:
                    fn_classes[expected_label] = fn_classes[expected_label] + 1

            except AssertionError:
                continue

    return tp_classes, fp_classes, fn_classes, count_classes


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

    data_accuracy = model.get_data(model.get_accuracy_csv())
    data_precision = model.get_data(model.get_precision_csv())
    data_recall = model.get_data(model.get_recall_csv())
    data_f1 = model.get_data(model.get_f1_csv())

    ckpts = get_all_checkpoints_for_model(model)
    labels = model.read_labels()
    start_ckpt = model.get_ckpt_start()

    for ckpt in ckpts:
        if ckpt > start_ckpt:

            tp_classes, fp_classes, fn_classes, count_classes = predict(
                model, ckpt)

            series_accuracy = pd.Series(dtype=np.float64)
            series_precision = pd.Series(dtype=np.float64)
            series_recall = pd.Series(dtype=np.float64)
            series_f1 = pd.Series(dtype=np.float64)

            for label in labels:

                tp = tp_classes[label]

                accuracy = tp / count_classes[label]
                precision = tp / (tp + fp_classes[label])
                recall = tp / (tp + fn_classes[label])
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = (2 * precision * recall) / (precision + recall)

                series_accuracy = series_accuracy.set_value(label, accuracy)
                series_precision = series_precision.set_value(label, precision)
                series_recall = series_recall.set_value(label, recall)
                series_f1 = series_f1.set_value(label, f1)

            data_accuracy[str(ckpt)] = series_accuracy
            data_precision[str(ckpt)] = series_precision
            data_recall[str(ckpt)] = series_recall
            data_f1[str(ckpt)] = series_f1

            model.write_ckpt_start(ckpt)

            model.save_data(data_accuracy, model.get_accuracy_csv())
            model.save_data(data_precision, model.get_precision_csv())
            model.save_data(data_recall, model.get_recall_csv())
            model.save_data(data_f1, model.get_f1_csv())

            print("---")
            print("Accuracy:")
            print(data_accuracy)
            print("---")
            print("Precision:")
            print(data_precision)
            print("---")
            print("Recall:")
            print(data_recall)
            print("---")
            print("F1:")
            print(data_f1)


if __name__ == '__main__':

    for i in range(0, 9):
        process_model(Model(i+1))
