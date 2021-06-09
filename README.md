# Table of contents
-  [Introduction](#introduction)
-  [Requirements](#requirements)
-  [Downloading custom YOLOv4 weights](#Downloading-custom-YOLOv4-weights)
-  [Training model](#training-model)
-  [Testing model](#testing-model)
-  [Testing model with SemEval scripts](#testing-model-with-semeval-scripts)
-  [Project structure](#project-structure)
-  [Pretrained RoBERTa](#pretrained-roberta)
-  [Credits](#credits)

## Introduction
Object tracking implemented with YOLOv4 (Keras), DeepSort (TensorFlow). YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker. Above pipeline was used on PAMELA-UANDES DATASET in order to detect, track and count people getting on and off a metropolitan train. 

## Requirements
```bash
pip install -r requirements.txt
```

## Downloading custom YOLOv4 weights
Our object tracker uses YOLOv4 to make the object detections, which deep sort then uses to track. There exists an official pre-trained YOLOv4 object detector model that is able to detect 80 classes. Unfortunately default detection fails when used on videos from PAMELA-UANDES DATASET. In order to fix that problem custom YOLO model had to be trained. Entire procedure is described here. 

Download custom yolov4-custom_best.weights file: https://drive.google.com/file/d/16rHphzm1DD0wivlq7yxn1U8AounwHQN6/view?usp=sharing

Copy and paste yolov4-custom_best.weights from your downloads folder into the 'data' folder of this repository.

## Running the Tracker with YOLOv4
All we need to do is run the object_tracker.py script to run our object tracker with YOLOv4 and DeepSort.
```bash

# Run yolov4 deep sort object tracker on video and print tracks statistics
python object_tracker.py --output outputs/output.avi --info

```
The output flag allows you to save the resulting video of the object tracker running so that you can view it again later. Video will be saved to the path that you set. (outputs folder is where it will be if you run the above command!)

As mentioned above, the resulting video will save to wherever you set the ``--output`` command line flag path to. I always set it to save to the 'outputs' folder. You can also change the type of video saved by adjusting the ``--output_format`` flag, by default it is set to AVI codec which is XVID.

Example video showing tracking of all coco dataset classes:
<p align="center"><img src="data/helpers/all_classes.gif"\></p>

## Command Line Args Reference

```bash
       USAGE: object_tracker.py [flags]
flags:

object_tracker.py:
  --[no]count: count objects being tracked on screen
    (default: 'false')
  --[no]dont_show: dont show video output
    (default: 'false')
  --[no]info: show detailed info of tracked objects
    (default: 'false')
  --output: path to output video
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID')
  --video: path to input video or set to 0 for webcam
    (default: './data/video/test5.mpg')
  --weights: path to weights file
    (default: 'data/yolov4-custom_best.weights')
```

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)

### To do:
- dorobić logikę aplikacji zliczającą wchodzących i wychodzących ludzi 
- evaluacja śledzenia (MOTA)
- 
