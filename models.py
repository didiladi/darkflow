
PATH_TINY_YOLO_VOC_WEIGHTS = '../weights/tiny-yolo-voc.weights'

PATH_CFG_TINY_YOLO_NEW_1 = 'cfg/tiny-yolo-voc-v2-1.cfg'
PATH_CFG_TINY_YOLO_NEW_2 = 'cfg/tiny-yolo-voc-new-2.cfg'
PATH_CFG_TINY_YOLO_NEW_3 = 'cfg/tiny-yolo-voc-new-3.cfg'

PATH_IMAGES_TRAIN_1 = "../data/images/train-v2-1"
PATH_IMAGES_TRAIN_2 = "../data/images/train-2"
PATH_IMAGES_TRAIN_3 = "../data/images/train-3"

PATH_IMAGES_DEV_1 = "../data/images/dev-v2-1"
PATH_IMAGES_DEV_2 = "../data/images/dev-2"
PATH_IMAGES_DEV_3 = "../data/images/dev-3"

PATH_IMAGES_TEST_1 = "../data/images/test-v2-1"
PATH_IMAGES_TEST_2 = "../data/images/test-2"
PATH_IMAGES_TEST_3 = "../data/images/test-3"

PATH_ANNOTATIONS_TRAIN_1 = "../data/images/train-annotations-v2-1"
PATH_ANNOTATIONS_TRAIN_2 = "../data/images/train-annotations-2"
PATH_ANNOTATIONS_TRAIN_3 = "../data/images/train-annotations-3"

PATH_ANNOTATIONS_DEV_1 = "../data/images/dev-annotations-v2-1"
PATH_ANNOTATIONS_DEV_2 = "../data/images/dev-annotations-2"
PATH_ANNOTATIONS_DEV_3 = "../data/images/dev-annotations-3"

PATH_ANNOTATIONS_TEST_1 = "../data/images/test-annotations-v2-1"
PATH_ANNOTATIONS_TEST_2 = "../data/images/test-annotations-2"
PATH_ANNOTATIONS_TEST_3 = "../data/images/test-annotations-3"

PATH_LABELS_1 = "labels-v2-1.txt"
PATH_LABELS_2 = "labels-2.txt"
PATH_LABELS_3 = "labels-3.txt"


class Model(object):
    """ The abstract base of our models """

    def get_path_for_train_images(self):
        raise NotImplementedError("Should have implemented this")

    def get_path_for_dev_images(self):
        raise NotImplementedError("Should have implemented this")

    def get_path_for_test_images(self):
        raise NotImplementedError("Should have implemented this")

    def get_path_for_train_annotations(self):
        raise NotImplementedError("Should have implemented this")

    def get_path_for_dev_annotations(self):
        raise NotImplementedError("Should have implemented this")

    def get_path_for_test_annotations(self):
        raise NotImplementedError("Should have implemented this")

    def get_config(self):
        raise NotImplementedError("Should have implemented this")

    def get_weights(self):
        return PATH_TINY_YOLO_VOC_WEIGHTS

    def get_labels(self):
        raise NotImplementedError("Should have implemented this")


class Model1(Model):
    """ The first model """

    def get_path_for_train_images(self):
        return PATH_IMAGES_TRAIN_1

    def get_path_for_dev_images(self):
        return PATH_IMAGES_DEV_1

    def get_path_for_test_images(self):
        return PATH_IMAGES_TEST_1

    def get_path_for_train_annotations(self):
        return PATH_ANNOTATIONS_TRAIN_1

    def get_path_for_dev_annotations(self):
        return PATH_ANNOTATIONS_DEV_1

    def get_path_for_test_annotations(self):
        return PATH_ANNOTATIONS_TEST_1

    def get_config(self):
        return PATH_CFG_TINY_YOLO_NEW_1

    def get_labels(self):
        return PATH_LABELS_1


class Model2(Model):
    """ The second model """

    def get_path_for_train_images(self):
        return PATH_IMAGES_TRAIN_2

    def get_path_for_dev_images(self):
        return PATH_IMAGES_DEV_2

    def get_path_for_test_images(self):
        return PATH_IMAGES_TEST_2

    def get_path_for_train_annotations(self):
        return PATH_ANNOTATIONS_TRAIN_2

    def get_path_for_dev_annotations(self):
        return PATH_ANNOTATIONS_DEV_2

    def get_path_for_test_annotations(self):
        return PATH_ANNOTATIONS_TEST_2

    def get_config(self):
        return PATH_CFG_TINY_YOLO_NEW_2

    def get_labels(self):
        return PATH_LABELS_2


class Model3(Model):
    """ The third model """

    def get_path_for_train_images(self):
        return PATH_IMAGES_TRAIN_3

    def get_path_for_dev_images(self):
        return PATH_IMAGES_DEV_3

    def get_path_for_test_images(self):
        return PATH_IMAGES_TEST_3

    def get_path_for_train_annotations(self):
        return PATH_ANNOTATIONS_TRAIN_3

    def get_path_for_dev_annotations(self):
        return PATH_ANNOTATIONS_DEV_3

    def get_path_for_test_annotations(self):
        return PATH_ANNOTATIONS_TEST_3

    def get_config(self):
        return PATH_CFG_TINY_YOLO_NEW_3

    def get_labels(self):
        return PATH_LABELS_3
