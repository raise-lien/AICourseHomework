# coding=utf-8
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import sys

reload(sys)  
sys.setdefaultencoding('gbk') 

x1_float = np.arange(-5.00, 5.01, 0.01)
x2_float = np.arange(-5.00, 5.01, 0.01)
print x1_float
print x2_float


def tanh_activation(x):
    a = 1.716
    b = (2.0 / 3.0)
    e_p = np.exp(b * x)
    e_m = np.exp(b * (-x))
    return a * ((e_p - e_m) / (e_p + e_m))


def get_matrix(m, n):
    mat = []
    for i in range(m):
        mat.append([0.0] * n)
    return mat


def network_inference(input_n, hidden_n, output_n, weight_ih, weight_ho, input_data):
    input_cells = [1.0] * (input_n + 1)
    hidden_cells = [1.0] * (hidden_n + 1)
    output_cell = [1.0] * output_n
    for i in range(input_n):
        input_cells[i] = input_data[i]
    for j in range(hidden_n):
        total = 0.0
        for i in range(input_n + 1):
            total += input_cells[i] * weight_ih[i][j]
        hidden_cells[j] = tanh_activation(total)
    for k in range(output_n):
        total = 0.0
        for j in range(hidden_n + 1):
            total += hidden_cells[j] * weight_ho[j][k]
        output_cell[k] = tanh_activation(total)
    return output_cell


if __name__ == '__main__':
    weight_in_hidden = get_matrix(3, 2)
    weight_hidden_out = get_matrix(3, 1)

    weight_in_hidden[0][0] = 0.5
    weight_in_hidden[0][1] = -0.5
    weight_in_hidden[1][0] = 0.3
    weight_in_hidden[1][1] = -0.4
    weight_in_hidden[2][0] = -0.1
    weight_in_hidden[2][1] = 1.0

    weight_hidden_out[0][0] = 1.0
    weight_hidden_out[1][0] = -2.0
    weight_hidden_out[2][0] = 0.5

    x1_r = []
    x2_r = []
    x1_g = []
    x2_g = []
    for i in range(1001):
        for j in range(1001):
            data = [x1_float[i], x2_float[j]]
            predict = network_inference(
                2, 2, 1, weight_in_hidden, weight_hidden_out, data)
            if predict[0] > 0.0:
                x1_r.append(data[0])
                x2_r.append(data[1])
            else:
                x1_g.append(data[0])
                x2_g.append(data[1])
    plt.figure(1)
    plt.scatter(x1_r, x2_r, c='w')
    plt.scatter(x1_g, x2_g, c='k')
    plt.savefig('params_1.png')

    # change the weight
    c_x1_r = []
    c_x2_r = []
    c_x1_g = []
    c_x2_g = []

    weight_in_hidden[0][0] = -1.0
    weight_in_hidden[0][1] = 1.0
    weight_in_hidden[1][0] = -0.5
    weight_in_hidden[1][1] = 1.5
    weight_in_hidden[2][0] = 1.5
    weight_in_hidden[2][1] = -0.5

    weight_hidden_out[0][0] = 0.5
    weight_hidden_out[1][0] = -1.0
    weight_hidden_out[2][0] = 1.0

    for i in range(1001):
        for j in range(1001):
            data = [x1_float[i], x2_float[j]]
            predict = network_inference(
                2, 2, 1, weight_in_hidden, weight_hidden_out, data)
            # print predict
            if predict[0] > 0.0:
                c_x1_r.append(data[0])
                c_x2_r.append(data[1])
            else:
                c_x1_g.append(data[0])
                c_x2_g.append(data[1])
    plt.figure(2)
    plt.scatter(c_x1_r, c_x2_r, c='w')
    plt.scatter(c_x1_g, c_x2_g, c='k')
    plt.savefig('params_2.png')
    plt.show()
