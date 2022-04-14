# from mmdet.apis import init_detector
# from mmdet.apis import inference_detector
# from mmdet.apis import show_result

# 模型配置文件
config_file = 'configs/yolox/yolox_tiny_8x8_300e_coco.py'

# 预训练模型文件
checkpoint_file = 'yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

# 通过模型配置文件与预训练文件构建模型
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并进行展示
img = 'BUPT1.png'
# img = 'demo/demo.jpg'
# result = inference_detector(model, img)
# show_result(img, result, model.CLASSES)

import mmcv
import cv2
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import matplotlib.pyplot as plt

model = init_detector(config_file,checkpoint_file)

result = inference_detector(model,img)

show_result_pyplot(model, img, result, score_thr=0.6)# show the image with result
model.show_result(img, result, out_file='bbb.jpg')# save image with result