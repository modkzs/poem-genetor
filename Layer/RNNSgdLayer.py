# -*- coding: utf-8 -*-
from . import PlainLayer
import numpy as np
from .Function import function
__author__ = 'yixuanhe'


class RNNSgdLayer(PlainLayer.PlainLayer):
    """RNN layer in neural network using sgd.
    This layer can only be used as hidden layer
    because only hidden layer need to get the
    output value of last layer as input"""

    def __init__(self, cell_num, func, input_num, backoff):

        self.cell_num = cell_num
        self.func = func
        self.input_num = input_num + 1
        weight = []
        for i in range(self.cell_num):
            weight.append(np.random.normal(loc=0.0, scale=1.0, size=self.input_num))
        self.input_weight = np.array(weight)

        weight = []
        for i in range(self.cell_num):
            weight.append(np.random.normal(loc=0.0, scale=1.0, size=self.cell_num))
        self.layer_weight = np.array(weight)

        self.backoff = backoff

        # s and x store history input:
        # s : St-4, St-3, St-2, St-1
        # x : Xt-3, Xt-2, Xt-1, Xt
        self.s = []
        self.x = []

        for i in range(backoff):
            self.s.append(0)
            self.x.append(0)

    @staticmethod
    def _push_value(l, value):
        """This is a pop operation of queue implemented by list"""
        l.pop(1)
        l.append(value)

    def computeNet(self, input, layer):
        input = np.append(input, [1])

        self.input = input
        self._push_value(self.x, input)
        self.layer = layer
        self._push_value(self.s, layer)

        self.net = np.dot(self.input_weight, input)
        self.net += np.dot(self.layer_weight, layer)

    def update(self):
        self.layer_weight -= self.derivative_layer
        self.input_weight -= self.derivative_input


    def computeOutput(self):
        value = []

        for n in self.net:
            value.append(self.func.f(n))
        return np.array(value)

    def getDerivative(self, err):
        derivative_input = []
        derivative_layer = []
        last = []
        for i in range(self.input_num):
            d_i = self.func.derivative(self.input, self.input_weight[i])
            delta = err[i] * d_i
            derivative_input.append(delta)

        for i in range(self.cell_num):
            derivative_layer.append(0)
            last.append(1)

        for i in range(-1, -self.backoff-1, -1):
            for j in range(self.cell_num):
                d_l = self.func.derivative(self.s[i], self.layer_weight[j])
                delta = err[j] * d_l
                last[j] *= delta * self.s[i]
                derivative_layer[j] += last[j]

        self.derivative_input = np.array(derivative_input).T
        self.derivative_layer = np.array(derivative_layer)

    def getDelte(self):
        pass


if __name__ == '__main__':

    sg = function.Sigmoid()
    layer = RNNSgdLayer(2, sg, 5, 4)

    x = np.random.normal(loc=0.0, scale=1.0, size=5)
    l = np.random.normal(loc=0.0, scale=1.0, size=2)
    y = np.array([1, 2])

    layer.computeNet(x, l)
    layer.computeOutput()

    layer.getDerivative(y)

