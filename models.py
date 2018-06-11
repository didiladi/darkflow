import numpy as np
import pandas as pd
import subprocess
import os
import time
import json
import os.path

PATH_TINY_YOLO_VOC_WEIGHTS = '../weights/tiny-yolo-voc.weights'

PATH_CFG_PREFIX = 'cfg/tiny-yolo-voc-v2-'
PATH_CFG_SUFFIX = '.cfg'

PATH_TRAIN_DATA_PREFIX = '../data/images/train-v2-'
PATH_TEST_DATA_PREFIX = '../data/images/test-v2-'

PATH_TRAIN_DATA_ANNOTATIONS_PREFIX = '../data/images/train-annotations-v2-'
PATH_TEST_DATA_ANNOTATIONS_PREFIX = '../data/images/test-annotations-v2-'

PATH_LABELS_PREFIX = "labels-v2-"
PATH_LABELS_SUFFIX = ".txt"


class Model(object):

    def __init__(self, num):
        self.num = str(num)

    def get_path_for_train_images(self):
        return PATH_TRAIN_DATA_PREFIX + self.num

    def get_path_for_test_images(self):
        return PATH_TEST_DATA_PREFIX + self.num

    def get_path_for_train_annotations(self):
        return PATH_TRAIN_DATA_ANNOTATIONS_PREFIX + self.num

    def get_path_for_test_annotations(self):
        return PATH_TEST_DATA_ANNOTATIONS_PREFIX + self.num

    def get_config(self):
        return PATH_CFG_PREFIX + self.num + PATH_CFG_SUFFIX

    def get_csv(self):
        return self.get_config() + ".csv"

    def get_labels(self):
        return PATH_LABELS_PREFIX + self.num + PATH_LABELS_SUFFIX

    def get_start_weights(self):
        return PATH_TINY_YOLO_VOC_WEIGHTS

    def get_ckpt_prefix(self):

        config_path = self.get_config()

        # it has this format: 'cfg/tiny-yolo-voc-new-1.cfg'
        return config_path.split("/")[1].split(".")[0]

    def get_ckpt_start(self):

        prefix = self.get_ckpt_prefix()

        try:

            label_file = open("cfg/" + prefix + ".ckpt-start", 'r')
            lines = label_file.readlines()
            label_file.close()

            return int(lines[0])

        except IOError:
            return 0

        return 0

    def write_ckpt_start(self, ckpt):

        prefix = self.get_ckpt_prefix()

        text_file = open("cfg/" + prefix + ".ckpt-start", "w")
        text_file.write(str(ckpt))
        text_file.close()

    def read_labels(self):
        """ Loads the labels of the imagenet synsets which should be downloaded """

        print("Reading desired labels")

        label_file = open(self.get_labels(), 'r')
        lines = label_file.readlines()
        result = []

        for line in lines:
            result.append(line.rstrip('\n'))

        label_file.close()

        return result

    def get_evaluation_data(self):
        ''' 
        Returns a pandas data frame, which contains the CSV file (current 
        evaluation result) of the given model 
        '''

        file_path = self.get_csv()

        if os.path.isfile(file_path):
            return pd.read_csv(file_path, sep=',',
                               encoding='utf-8', index_col=0)

        labels = self.read_labels()
        return pd.DataFrame(data=None, index=labels, dtype=np.float64)

    def save_evaluation_data(self, data):
        data.to_csv(self.get_csv(), sep=',', encoding='utf-8')
