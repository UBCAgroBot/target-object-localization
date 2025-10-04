## Target Object Localization

**Details:** this project involves designing an object localizer using Pytorch. The main approach will be to converge to a RCNN approach, with the option of pivoting to YOLOv8 at the end if that is of interest. The dataset will consist of the following: humans, rocks, animals, and miscellaneous. In general, we want data similar to the ones we'll be expecting to get off of the front-facing camera.

**Team size:** 3 

**Progression:** 
- creating a basic CNN for object detection
- using basic sliding window trick to get bounding boxes (naive) 
- then using segmentation to get regions of interest for optimization 
- upgrading RCNN to Fast RCNN 
- upgrading Fast RCNN to Faster RCNN
- gathering dataset, cleaning and preprocessing
- training model + hyperparameter search

**Technologies:** Pytorch, Python, (maybe a little bit of W&B if we have time at the end)
