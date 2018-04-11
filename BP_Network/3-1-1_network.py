# coding=utf-8

import math
import random
import xlrd
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def get_matrix(I, J):
    m = []
    for i in range(I):
        m.append([0.0] * J)
    return m


def randomizeMatrix(matrix, a, b):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)


def set_Matrix(matrix, a):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = a


def tanh_activation(x):
    return float(1.716 * math.tanh(2.0 * x / 3.0))


def d_tanh_activation(y):
    return (1.716 * 2.0 / 3.0) * (1.0 - math.tanh(y) ** 2)
    # return (1.0 - math.tanh(y) ** 2)


class NN:
    def __init__(self, ni, nh, no, w_is_set):
        self.ni = ni + 1
        self.nh = nh + 1
        self.no = no
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no
        self.wi = get_matrix(self.ni, self.nh - 1)
        self.wo = get_matrix(self.nh, self.no)
        if w_is_set:
            set_Matrix(self.wi, 0.5)
            set_Matrix(self.wo, -0.5)
        else:
            randomizeMatrix(self.wi, -1.0, 1.0)
            randomizeMatrix(self.wo, -1.0, 1.0)

    def runNN(self, inputs):
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]
        for j in range(self.nh - 1):
            sum = 0.0
            for i in range(self.ni):
                sum += (self.ai[i] * self.wi[i][j])
            self.ah[j] = tanh_activation(sum)
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += (self.ah[j] * self.wo[j][k])
            self.ao[k] = tanh_activation(sum)
        return self.ao

    def backPropagate(self, labels, N):
        # 计算输出层 deltas
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = labels[k] - self.ao[k]
            output_deltas[k] = error * d_tanh_activation(self.ao[k])
        # 更新输出层权值
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += change * N
        # 计算隐藏层 deltas
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * d_tanh_activation(self.ah[j])

        # 更新输入层权值
        for i in range(self.ni):
            for j in range(self.nh - 1):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += change * N
        # 计算误差平方和
        error = 0.0
        for k in range(len(labels)):
            error = 0.5 * (labels[k] - self.ao[k]) ** 2
        return error

    def weights(self):
        print 'Input weights:'
        for i in range(self.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.nh):
            print self.wo[j]
        print ''

    def test(self, patterns):
        p_n = 0
        n_n = 0
        for p in patterns:
            inputs = p[0]
            outputs = self.runNN(inputs)
            if outputs[0] <= 0.5 and p[1][0] == 0.1:
                print "n", outputs[0], p[1][0]
                n_n += 1
            elif outputs[0] > 0.5 and p[1][0] == 1.0:
                print "p", outputs[0], p[1][0]
                p_n += 1
            print 'Inputs:', p[0], '-->', outputs, '\tTarget', p[1]
        print "Acc: ", (n_n + p_n) / 20.00

    def train(self, data_input, max_iterations=10000, N=0.1):
        indexs = np.arange(20)
        print indexs
        random.shuffle(indexs)
        epoch_error = []
        index = indexs[0]
        k = 0
        for i in range(max_iterations):
            error = 0.0
            p = data_input[index]
            if k == 19:
                k = 0
                random.shuffle(indexs)
                index = indexs[k]
            else:
                k += 1
                index = indexs[k]
            inputs = p[0]
            labels = p[1]
            self.runNN(inputs)
            error += self.backPropagate(labels, N)
            print 'Combined error', error
            epoch_error.append(error)
            if error < 0.00005:
                break
        self.test(data_input)
        return epoch_error, i + 1


def main():
    w_is_set = True
    iter_n = 10000
    myNN = NN(3, 1, 1, w_is_set)
    data = xlrd.open_workbook('3_d_data.xlsx')
    table = data.sheets()[0]
    nrows = table.nrows
    pat = []
    for i in range(nrows):
        pat.append([table.row_values(i)[:3], [0.1]])
        pat.append([table.row_values(i)[3:6], [1.0]])
    iter_error, new_iter_n = myNN.train(data_input=pat, max_iterations=iter_n)
    train_iter = np.arange(new_iter_n)
    myNN.weights()
    plt.plot(train_iter, iter_error)
    plt.xlim((-1, new_iter_n + 1))
    plt.ylim((0, np.max(iter_error)+1))
    plt.xlabel('Epoch Number')
    plt.ylabel('Train Epoch Error')
    if w_is_set:
        plt.title('Training Loss Graph, initial weight is -+0.5')
        plt.savefig('loss_weight_0.5.png')
    else:
        plt.title('Training Loss Graph, initial weight is random (-1, 1)')
        plt.savefig('loss_weight_1.png')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
