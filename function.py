# -*- coding: utf-8 -*-
__author__ = 'yixuanhe'

from abc import ABCMeta, abstractmethod


class Function(metaclass=ABCMeta):
    """This is a function, f is ths function and derivative"""

    @abstractmethod
    def f(self, x1, x2):
        pass

    @abstractmethod
    def derivative(self, x1, x2):
        pass


class cross_entropy(Function):

    def f(self, y, z):
        pass

    def derivative(self, y, z):
        pass


class sigmoid(Function):

    def f(self, x, w):
        pass

    def derivative(self, x, w):
        pass


class softplus(Function):

    def f(self, x, w):
        pass

    def derivative(self, x, w):
        pass