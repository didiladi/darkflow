#!/bin/sh

./flow --model cfg/tiny-yolo-voc-new-3.cfg --train --dataset ../data/images/train-3 --annotation ../data/images/train-annotations-3 --labels labels-3.txt --load ../weights/tiny-yolo-voc.weights
# ./flow --model cfg/tiny-yolo-voc-new-3.cfg --train --dataset ../data/images/train-3 --annotation ../data/images/train-annotations-3 --labels labels-3.txt --load -1
