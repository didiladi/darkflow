#!/bin/sh

./flow --model cfg/tiny-yolo-voc-new-1.cfg --train --dataset ../data/images/train-1 --annotation ../data/images/train-annotations-1 --load ../weights/tiny-yolo-voc.weights
# ./flow --model cfg/tiny-yolo-voc-new-1.cfg --train --dataset ../data/images/train-1 --annotation ../data/images/train-annotations-1 --load -1
