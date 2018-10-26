#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/10/26 下午10:44

import logging

LOG_FORMAT = "[%(asctime)s] [%(processName)s:%(threadName)s] [%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
Logger = logging.getLogger(__name__)


class Demo(object):

    def __init__(self):
        pass

    def step1(self):
        Logger.debug('step1')

    def step2(self):
        Logger.debug('step2')

    def run(self):
        self.step1()
        self.step2()


def main():
    Demo().run()


if __name__ == '__main__':
    main()