# /usr/bin/python
# -*- encoding:utf-8 -*-

from PIL import Image
import numpy as np
import pandas as pd
import os

pic_file_path = 'E:/Tianchi/TB_data/tianchi_fm_img2_1'

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

def basic_op(path):
    pil_im = Image.open(path)
    # 转换为灰度图像
    pil_im_gray = pil_im.convert('L')
    # 创建缩略图
    pil_thumbnail = pil_im.thumbnail((128, 128))
    # 裁剪指定区域
    box = (100, 100, 400, 400)
    pil_box = pil_im.crop(box)

