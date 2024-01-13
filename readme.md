# [WIP] Parkspace detection using YOLOv5 OpenCV DNN
## Overview
Parkspace detection using YOLOv5 (An object detection model built on PyTorch)
on OpenCV DNN module (a single API for performing Deep Learning inference and has very few dependencies)

## Result


https://github.com/saviogeorge/parkspace_detection/assets/20711873/f53f23eb-4215-40ab-b307-0fde49d263d2

## TODO

1. ROI based processing, also handle the case when no vehicles are detected parked.
2. Consider IoU with a threshold instead of checking for bounding box intersection.
3. Calculate if a car fit in when a space is detected. 

## References

1. [Object detection using YOLO source](https://github.com/spmallick/learnopencv/tree/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python)
2. [Object detection using YOLO](https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/)
3. https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400
4. https://stackoverflow.com/questions/42333338/opencv-build-g-symbol-error-dso-error
5. https://docs.opencv.org/4.x/d4/db9/samples_2dnn_2object_detection_8cpp-example.html
6. https://docs.opencv.org/4.x/d1/d0a/classcv_1_1dnn__objdetect_1_1InferBbox.html#a480ff4d0ebdb8a41e685e3f719b8e607
7. https://github.com/CheckBoxStudio/IoU
8. https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/





