#!/bin/sh

./flow --model cfg/tiny-yolo-voc-new-2.cfg --train --dataset ../data/images/train-2 --annotation ../data/images/train-annotations-2 --labels labels-2.txt --load ../weights/tiny-yolo-voc.weights
#./flow --model cfg/tiny-yolo-voc-new-2.cfg --train --dataset ../data/images/train-2 --annotation ../data/images/train-annotations-2 --labels labels-2.txt --load -1
