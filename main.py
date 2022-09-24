#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-09-24 17:14
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    main.py
# @Project: paddle-2.3.2-features
# @Package: 
# @Ref:
import paddle

paddle.fluid.install_check.run_check()

paddle.fluid.is_compiled_with_cuda()

