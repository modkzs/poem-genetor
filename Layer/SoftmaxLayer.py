# -*- coding: utf-8 -*-
from . import PlainLayer
import numpy as np
from .Function import Softmax
__author__ = 'yixuanhe'


class SoftmaxLayer(PlainLayer.PlainLayer):

    def __init__(self, output_num, input_num):
        self.output_num = output_num
        self.func = Softmax.Softmax()
        self.input_num = input_num

        weight = []
        for i in range(output_num):
            weight.append(np.random.normal(loc=0.0, scale=1.0, size=self.input_num))

        self.weight = np.array(weight)

    def update(self):
        self.weight -= self.derivative

    def getDerivative(self, y):
        derivative = []
        delta = []

        for i in range(self.output_num):
            delta.append(self.func.derivative(y[i], self.value[i]))
            derivative.append(delta[i]*self.input)

        self.delta = delta
        self.derivative = np.array(derivative)
        return self.derivative

    def computeNet(self, x):
        self.net = np.dot(self.weight, x)
        self.input = x

    def computeOutput(self):
        value = self.func.f(self.net)
        self.value = np.array(value)
        return self.value

    def getActive(self):
        value = self.computeOutput()
        max = 0
        for i in range(self.output_num):
            if value[i] > value[max]:
                max = i

        return max

    def getDelte(self):
        return self.delta