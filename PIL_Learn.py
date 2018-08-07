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
    print(pil_im)
    pil_im.show()
    # 转换为灰度图像
    pil_im_gray = pil_im.convert('L')
    pil_im_gray.show()
    # 创建缩略图, 保持比例边长不超过设置
    pil_im.thumbnail((128, 128))
    print(pil_im)
    pil_im.show()
    # 裁剪指定区域
    box = (100, 100, 400, 400)
    pil_box = pil_im.crop(box)
    pil_box = pil_box.transpose(Image.ROTATE_180)
    pil_im.paste(pil_box, box)


if __name__ == '__main__':
    imlist = get_imlist(pic_file_path)
    basic_op(imlist[0])
