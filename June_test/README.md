# 說明
## 環境需求
```
!pip install tensorflow-gpu 
!pip install scipy 
!pip install pyyaml 
!pip install opencv-python==3.4.5.20
%tensorflow_version 1.x 
```

## 引用函式庫
```
from IPython import display
from PIL import Image
import numpy as np
import tensorflow as tf
import sys
import coco_metric
from mask_rcnn.object_detection import visualization_utils
import matplotlib.pyplot as plt
import cv2
```

## 程序簡述
- horse_imgseg_pose.ipynb
    - 單純把maskrcnn和poseNet接起來
- image_horse_imgseg_pose.ipynb
    - 定義函數，接入影像
- ver2_image_horse_imgseg_pose.ipynb
    - 對整個資料夾的圖像進行人體的語意分割後，儲存到資料夾(share_folder中)
    - 增加語意分割的定界框的可信度檢驗
    - 對整個share_folder的所有包含人體的圖像做姿態辨識
        - Issue: 阮同學自己拍的圖片可以進行分割，但不能辨識資態，應該是解析度太高