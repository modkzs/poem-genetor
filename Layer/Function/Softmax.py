# -*- coding: utf-8 -*-
from . import function
import numpy as np
__author__ = 'yixuanhe'


class Softmax(function.Function):
    def f(self, nets, x2=0):
        value = []
        sum = 0
        for x in nets:
            v = np.exp(x)
            value.append(v)
            sum += v

        for i in range(len(value)):
            value[i] = value[i]/sum

        return value

    def derivative(self, y, v):
        return y - y*v
