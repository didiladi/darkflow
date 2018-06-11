from darkflow.net.build import TFNet
import cv2

options = {
    "model": "cfg/tiny-yolo-voc-v2-1.cfg",
    "metaLoad": "built-graph/tiny-yolo-voc-v2-1.meta",
    "pbLoad": "built-graph/tiny-yolo-voc-v2-1.pb",
    "threshold": 0.13
}

tfnet = TFNet(options)

#imgcv = cv2.imread("./sample_img/sample_strawberry.jpg")
imgcv = cv2.imread("./sample_img/sample_barrow.jpg")
#imgcv = cv2.imread("./sample_img/sample_shopping_cart.jpg")
#imgcv = cv2.imread("./sample_img/sample_tractor.jpg")
#imgcv = cv2.imread("./sample_img/sample_fig.jpg")

result = tfnet.return_predict(imgcv)

print(result)
