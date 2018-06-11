
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

    def get_labels(self):
        return PATH_LABELS_PREFIX + self.num + PATH_LABELS_SUFFIX

    def get_start_weights(self):
        return PATH_TINY_YOLO_VOC_WEIGHTS
