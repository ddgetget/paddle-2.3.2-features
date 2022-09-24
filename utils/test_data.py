#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-09-24 17:48
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    test_data.py
# @Project: paddle-2.3.2-features
# @Package: 
# @Ref:
import pytest

from utils.data import get_insect_names, get_annotations, get_img_data_from_file

TRAINDIR = '../work/insects/train'
TESTDIR = '../work/insects/test'
VALIDDIR = '../work/insects/val'


@pytest.fixture
def label_list():
    print("初始化操作")
    pass


cname2cid = get_insect_names()
records = get_annotations(cname2cid, TRAINDIR)

print(len(records))
print(records[0])

record = records[0]


@pytest.mark.parametrize(
    "record",
    [(records[0]), (records[1]), (records[2])],
)
def test_get_img_data_from_file(label_list, record):
    label_list()
    img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
    # 函数可以返回图片数据的数据，它们是图像数据img，真实框坐标gt_boxes，真实框包含的物体类别gt_labels，图像尺寸scales。
    for key, value in zip(["img.shape", "gt_boxes.shape", "gt_labels", "scales"],
                          [img.shape, gt_boxes.shape, gt_labels, scales]):
        print(key, ":", value)
    # print("img.shape, gt_boxes.shape, gt_labels, scales", img.shape, gt_boxes.shape, gt_labels, scales)
