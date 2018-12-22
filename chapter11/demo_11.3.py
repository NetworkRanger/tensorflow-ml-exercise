#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午7:17

# 11.3 Tensorboard的进阶

# 从命令行下运行上面的脚本

"""
$ python3 using_tensorboard.py

Run the command: $ tensorbaord --logdir="tensorbaord" then navigate to http://127.0.0.1:6006
Generation 10 of 100. Train Loss: 20.4, Test Loss: 20.5.
Generation 20 of 100. Train Loss: 17.6, Test Loss: 20.5.
Generation 90 of 100. Train Loss: 20.1, Test Loss: 20.5.
Generation 100 of 100. Train Loss: 19.4, Test Loss: 20.5.
"""

# 然后运行指定的命令启动Tensorboard

"""
$ tensorboard --logdir="tensorboard"
Starting TensorBoard b'29' on port 6006
(You can navigate to http://127.0.0.1:6006)c
"""