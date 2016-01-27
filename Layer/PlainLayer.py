# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

__author__ = 'yixuanhe'


class PlainLayer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, cell_num, func, input_num):
        pass

    @abstractmethod
    def computeOutput(self):
        pass

    @abstractmethod
    def computeNet(self, x, y):
        pass

    @abstractmethod
    def getDerivative(self, err, x):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def getDelte(self):
        pass



