#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-09-24 17:57
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    test_data_process.py
# @Project: paddle-2.3.2-features
# @Package: 
# @Ref:
import pytest
from PIL import Image
import numpy as np
from utils.data import get_insect_names, get_annotations, get_img_data_from_file
from utils.data_process import random_distort, random_expand, random_crop, random_interp, random_flip, image_augment

TRAINDIR = '../work/insects/train'
TESTDIR = '../work/insects/test'
VALIDDIR = '../work/insects/val'
cname2cid = get_insect_names()
records = get_annotations(cname2cid, TRAINDIR)

# 定义可视化函数，用于对比原图和图像增强的效果
import matplotlib.pyplot as plt


def visualize(srcimg, img_enhance):
    # 图像可视化
    plt.figure(num=2, figsize=(6, 12))
    plt.subplot(1, 2, 1)
    plt.title('Src Image', color='#0000FF')
    plt.axis('off')  # 不显示坐标轴
    plt.imshow(srcimg)  # 显示原图片

    # 对原图做 随机改变亮暗、对比度和颜色等 数据增强
    srcimg_gtbox = records[0]['gt_bbox']
    srcimg_label = records[0]['gt_class']

    plt.subplot(1, 2, 2)
    plt.title('Enhance Image', color='#0000FF')
    plt.axis('off')  # 不显示坐标轴
    plt.imshow(img_enhance)
    plt.show()


@pytest.mark.parametrize(
    "record",
    [(records[0]), (records[1]), (records[2])],
)
def test_random_distort(record):
    image_path = records['im_file']
    print("read image from file {}".format(image_path))
    srcimg = Image.open(image_path)
    # 将PIL读取的图像转换成array类型
    srcimg = np.array(srcimg)
    # 对原图做 随机改变亮暗、对比度和颜色等 数据增强
    img_enhance = random_distort(srcimg)
    visualize(srcimg, img_enhance)


@pytest.mark.parametrize(
    "record",
    [(records[0]), (records[1]), (records[2])],
)
def test_random_expand(record):
    # 对原图做 随机改变亮暗、对比度和颜色等 数据增强
    srcimg_gtbox = record['gt_bbox']
    image_path = record['im_file']
    print("read image from file {}".format(image_path))
    srcimg = Image.open(image_path)
    # 将PIL读取的图像转换成array类型
    srcimg = np.array(srcimg)
    img_enhance, new_gtbox = random_expand(srcimg, srcimg_gtbox)
    visualize(srcimg, img_enhance)


@pytest.mark.parametrize(
    "record",
    [(records[0]), (records[1]), (records[2])],
)
def test_random_crop(record):
    # 对原图做 随机改变亮暗、对比度和颜色等 数据增强
    srcimg_gtbox = record['gt_bbox']
    srcimg_label = record['gt_class']

    image_path = record['im_file']
    print("read image from file {}".format(image_path))
    srcimg = Image.open(image_path)
    # 将PIL读取的图像转换成array类型
    srcimg = np.array(srcimg)
    img_enhance, new_labels, mask = random_crop(srcimg, srcimg_gtbox, srcimg_label)
    visualize(srcimg, img_enhance)


@pytest.mark.parametrize(
    "record,size",
    [(records[0], 640), (records[1], 650), (records[2], 20)],
)
def test_random_interp(record, size, interp=None):
    image_path = record['im_file']
    print("read image from file {}".format(image_path))
    srcimg = Image.open(image_path)
    # 将PIL读取的图像转换成array类型
    srcimg = np.array(srcimg)
    # 对原图做 随机改变亮暗、对比度和颜色等 数据增强
    img_enhance = random_interp(srcimg, size)
    visualize(srcimg, img_enhance)


@pytest.mark.parametrize(
    "record,gtboxes,thresh",
    [(records[0], 1, 0.4), (records[1], 1, 0.5), (records[2], 1, 0.6)],
)
def test_random_flip(record, gtboxes, thresh):
    image_path = record['im_file']

    print("read image from file {}".format(image_path))
    srcimg = Image.open(image_path)
    # 将PIL读取的图像转换成array类型
    srcimg = np.array(srcimg)
    srcimg_gtbox = record['gt_bbox']

    # 对原图做 随机改变亮暗、对比度和颜色等 数据增强
    img_enhance, box_enhance = random_flip(srcimg, srcimg_gtbox, thresh)
    visualize(srcimg, img_enhance)


@pytest.mark.parametrize(
    "record",
    [(records[0]), (records[1]), (records[2])],
)
def test_img_enhance(record):
    image_path = record['im_file']

    print("read image from file {}".format(image_path))
    srcimg = Image.open(image_path)
    # 将PIL读取的图像转换成array类型
    srcimg = np.array(srcimg)
    srcimg_gtbox = record['gt_bbox']
    srcimg_label = record['gt_class']

    img_enhance, img_box, img_label = image_augment(srcimg, srcimg_gtbox, srcimg_label, size=320)
    visualize(srcimg, img_enhance)


@pytest.mark.parametrize(
    "record",
    [(records[0]), (records[1]), (records[2])],
)
def test_data_frome_record(record):
    img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
    size = 512
    img, gt_boxes, gt_labels = image_augment(img, gt_boxes, gt_labels, size)
    print(img, gt_boxes, gt_labels)