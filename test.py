from mmdet.apis import init_detector, inference_detector

# 目标检测配置文件
config_file = 'configs/yolox/yolox_tiny_8x8_300e_coco.py'
# 训练模型
checkpoint_file = 'checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

# 配置模型
model = init_detector(config=config_file,
                      checkpoint=checkpoint_file,
                      device='cuda:0')

img = 'demo/demo.jpg'
#  推理实际调用语句
# results = model(return_loss=False, rescale=True, **data)
result = inference_detector(model=model, imgs=img)

# 打印结果
# print(model.CLASSES)
# for i in result:
#     print(i)

from PIL import Image, ImageDraw
# 打开原图
img = Image.open('demo/demo.jpg').convert('RGB')
# 画出目标框，因为一个类别可能对应多个目标
for item in result:
    for rec in item:
        x, y, w, h = rec[0], rec[1], rec[2], rec[3]
        draw = ImageDraw.Draw(img)
        draw.rectangle((x, y, w, h), width=2, outline='#41fc59')
# 保存结果图片
img.save('demo/result.png')