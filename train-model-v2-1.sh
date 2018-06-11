#!/bin/sh

# ./flow --model cfg/tiny-yolo-voc-v2-1.cfg --train --dataset ../data/images/train-v2-1 --annotation ../data/images/train-annotations-v2-1 --load ../weights/tiny-yolo-voc.weights
./flow --model cfg/tiny-yolo-voc-v2-1.cfg --train --dataset ../data/images/train-v2-1 --annotation ../data/images/train-annotations-v2-1 --load -1
