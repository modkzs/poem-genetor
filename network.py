# -*- coding: utf-8 -*-
from Layer import RNNSgdLayer, SoftmaxLayer
from Layer.Function.function import Sigmoid
import numpy as np
__author__ = 'yixuanhe'


class RNN:

    def __init__(self, output_num, hidden_num, input_num, backoff=4):
        self.output_num = output_num
        self.hidden_num = hidden_num
        self.input_num = input_num

        self.hidden_layer = RNNSgdLayer.RNNSgdLayer(self.hidden_num, Sigmoid(), input_num, backoff)
        self.output_layer = SoftmaxLayer.SoftmaxLayer(self.output_num, self.hidden_num)

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

    def save(self):
        np.save("model/out", self.output_layer.weight)
        np.save("model/hidden_input", self.hidden_layer.input_weight)
        np.save("model/hidden_layer", self.hidden_layer.layer_weight)

    def load(self):
        self.output_layer.weight = np.load("model/out.npy")
        self.hidden_layer.input_weight = np.load("model/hidden_input.npy")
        self.hidden_layer.layer_weight = np.load("model/hidden_layer.npy")



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


def getTraindata(data, number):
    datas = data.split(" ")
    pos = int(datas[1])+1
    features = datas[0]
    features_string = features.split(",")
    feature = []
    for f in features_string:
        if f != "":
            feature.append(float(f))
    feature = np.array(feature)

    label = []
    for i in range(number):
        if i != pos:
            label.append(0)
        else:
            label.append(1)
    label = np.array(label)

    return feature, label

if __name__ == "__main__":
    num = 7357
    rnn = RNN(num, 400, 301)

    n = 1
    for i in range(3):
        with open("/Volumes/devil/train_data") as f:
            for l in f:
                print(str(n) + "th train")
                n += 1
                x, y = getTraindata(l, num)
                rnn.train(x, y)

    rnn.save()
