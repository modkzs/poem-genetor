# -*- coding: utf-8 -*-
import PlainLayer
import numpy as np
from Function import function
__author__ = 'yixuanhe'


class BasicSgdLayer(PlainLayer.PlainLayer):
    """basic layer in neural network using sgd"""

    def __init__(self, cell_num, func, input_num):
        """
        Layer init. weight is input_num*output_num.
        Args:
            cell_num: the number of cell, also is output_num
            func: the active function
            input_num: the number of input feature

        Returns:
            null

        Raises:
            null
        """
        # we have a bias cell, so need 1 more cell
        self.cell_num = cell_num
        self.func = func
        self.input_num = input_num+1
        weight = []
        for i in range(self.cell_num):
            weight.append(np.random.normal(loc=0.0, scale=1.0, size=self.input_num))
        self.weight = np.array(weight)

    def computeNet(self, x):
        """The function which compute the value of net in this layer.
        This function is set to add batch normalization in some layer.
        Args:
            x: the input feature"""

        x = np.append(x, [1])
        self.input = x
        self.net = np.dot(self.weight, x)

    def computeOutput(self):
        """The function which compute the output in this layer.
        Return:
            value: the output value"""
        value = []

        for n in self.net:
            value.append(self.func.f(n))

        return np.array(value)

    def getDerivative(self, err):
        """The function which compute the derivative of this layer
        Args:
            err: the err in last layer
            x:   the input of this layer
        Returns
            derivative of this layer"""


        derivative = []
        delta = []
        for i in range(self.cell_num):
            d = self.func.derivative(self.input, self.weight[i])
            delt = err[i] * d
            delta.append(delt)
            derivative.append(delt * self.input)

        self.derivative = np.array(derivative)
        self.delta = np.array(delta)

        return self.derivative

    def update(self):
        """The function of updating weight.
        """

        self.weight -= self.derivative

    def getDelte(self):
        return self.delta

if __name__ == '__main__':
    sg = function.Sigmoid()
    layer = BasicSgdLayer(2, sg, 5)

    x = np.random.normal(loc=0.0, scale=1.0, size=5)
    l = np.random.normal(loc=0.0, scale=1.0, size=2)
    y = np.array([1, 2])

    layer.computeNet(x)
    layer.computeOutput()
    print(layer.getDerivative(y))

