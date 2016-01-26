# -*- coding: utf-8 -*-
__author__ = 'yixuanhe'

from abc import ABCMeta, abstractmethod
import numpy as np


class Function(metaclass=ABCMeta):
    """This is a function, f is ths function and derivative"""

    @abstractmethod
    def f(self, x1, x2):
        pass

    @abstractmethod
    def derivative(self, x1, x2):
        pass


class LMS(Function):
    """The Least Mean Square function, y is label computed by network, z is actual label.
    The derivative is f to y, of course"""

    def f(self, y, z):
        return np.dot((z-y).T, z-y)/2

    def derivative(self, y, z):
        return -(z-y)

class CrossEntropy(Function):
    """The cross entropy active function, w is weight, x is input.
    We ignore bias because x isaugmented matrix so w[-1] is bias.
    The derivative is f to w, of course"""

    def f(self, y, z=1):
        return z*np.log(y) + (1-z)*np.log(1-y)

    def derivative(self, y, z):
        return z/y - (1-z)/(1-y)


class Sigmoid(Function):
    """The sigmoid active function, w is weight, x is input.
    We ignore bias because x isaugmented matrix so w[-1] is bias.
    The derivative is f to w, of course"""

    def f(self, w, x=1):
        return 1/(1+np.exp(np.dot(w, x)))

    def derivative(self, x, w):
        v = 1/(1+np.exp(np.dot(w, x)))
        return v*(1-v)*x


class Softplus(Function):
    """The softplus active function, w is weight, x is input.
    We ignore bias because x is augmented matrix so w[-1] is bias.
    The derivative is f to w, of course"""

    def f(self, x, w=1):
        return np.log(1+np.exp(np.dot(w, x)))

    def derivative(self, x, w):
        v = np.exp(np.dot(w, x))
        return (v/(1+v))*x


if __name__ == '__main__':
    ww = np.array([1, 1, 1, 1])
    xx = np.array([2, 1, 3, 1])

    sp = Softplus()
    sg = Sigmoid()
    print(sp.derivative(xx.T, ww))
    print(sg.derivative(xx.T, ww))
    print(type(ww))
