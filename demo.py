# from mmdet.apis import init_detector
# from mmdet.apis import inference_detector
# from mmdet.apis import show_result
import mmcv
import cv2
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import matplotlib.pyplot as plt
# 模型配置文件
config_file = 'configs/yolox/yolox_tiny_8x8_300e_voc.py'
# config_file = 'configs/yolox/yolox_s_8x8_300e_voc_tfs.py'

# 预训练模型文件
checkpoint_file = '/data1/yuxinyi/project/search_yolox/tiny_tfs/epoch_300.pth'
# checkpoint_file = '/data1/yuxinyi/project/search_yolox/search_result1/epoch_300.pth'
model = init_detector(config_file, checkpoint_file)
# 通过模型配置文件与预训练文件构建模型
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并进行展示
for i in range(200):
    if i + 1 < 10:
        img = '/data0/public_data/VOCdevkit/VOC2007/JPEGImages/00000'+str(i+1)+'.jpg'
    elif i + 1 < 100:
        img = '/data0/public_data/VOCdevkit/VOC2007/JPEGImages/0000'+str(i+1)+'.jpg'
    elif i + 1 < 1000:
        img = '/data0/public_data/VOCdevkit/VOC2007/JPEGImages/000' + str(i + 1) + '.jpg'
    # result = inference_detector(model, img)
    # show_result(img, result, model.CLASSES)
    result = inference_detector(model,img)
    show_result_pyplot(model, img, result, score_thr=0.6)# show the image with result
    # model.show_result(img, result, out_file='/data1/yuxinyi/project/search_yolox/img/tfs/'+str(i+1)+'.jpg')# save image with result
    model.show_result(img, result, out_file='/data1/yuxinyi/project/search_yolox/img/tiny/'+str(i+1)+'.jpg')# save image with result