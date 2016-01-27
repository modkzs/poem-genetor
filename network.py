# -*- coding: utf-8 -*-
import Layer
import csv
__author__ = 'yixuanhe'


class RNN:

    def __init__(self, output_num, hidden_num, input_num, backoff=4):
        self.output_num = output_num
        self.hidden_num = hidden_num
        self.input_num = input_num

        self.hidden_layer = Layer.RNNSgdLayer(self.hidden_num, Layer.Function.Sigmoid(), input_num, backoff)
        self.output_layer = Layer.SoftmaxLayer(self.output_num, self.hidden_num)

        hidden_output = []
        for i in range(hidden_num):
            hidden_output.append(0)
        self.hidden_output = hidden_output

    def getOutput(self, x):
        self.hidden_layer.computeNet(x, self.hidden_output)
        output = self.hidden_layer.computeOutput()
        self.output_layer.computeNet(output)
        self.output_layer.computeOutput()

    def predict(self, x):
        self.getOutput(x)
        self.output_layer.getActive()

    def updateNetwork(self, y):
        delta = self.output_layer.getDerivative(y)
        self.output_layer.update()

        self.hidden_layer.getDerivative(delta)
        self.hidden_layer.update()

    def train(self, x, y):
        self.getOutput(x)
        self.updateNetwork(y)


def feature_generator(file_name, length):
    data_x = []
    data_y = []

    with open(file_name) as f:

        for l in f:
            data = l.split(" ")
            xs = data[0]
            ys = int(data[1])

            xi = xs.split(",")
            x = []
            for i in xi:
                x.append(float(i))

            y = []
            if ys == -1:
                ys = length
            for i in range(length):
                if i == ys:
                    y.append(1)
                else:
                    y.append(0)

            data_x.append(x)
            data_y.append(y)
    return data_x, data_y

if __name__ == "__main__":
    rnn = RNN(301, 400, 8000)

    x, y = feature_generator("data/train_data1", 100)

    leng = len(x)

    while True:
        for i in range(leng):
            rnn.train(x[i], y[i])

