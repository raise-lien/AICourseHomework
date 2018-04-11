import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.uniform(low=0.0, high=1.0, size=2000)
x2 = np.random.uniform(low=0.0, high=1.0, size=2000)
c1 = 0.0
c2 = 0.0


def cal_loss(x1_i, x2_i, x1_c, x2_c):
    return (x1_i - x1_c)**2 + (x2_i - x2_c)**2


def cal_pra_c1(x1_i, x1_c):
    return 2 * (x1_c - x1_i)


def cal_pra_c2(x2_i, x2_c):
    return 2 * (x2_c - x2_i)


def stochastic_gradient_descent_step(x1_p, x2_p, x1, x2):
    stepper = 0.001
    loss = 0.0
    x1_grad = 0
    x2_grad = 0
    data_x1 = x1
    data_x2 = x2
    x1_grad += cal_pra_c1(data_x1, x1_p) / 2
    x2_grad += cal_pra_c2(data_x2, x2_p) / 2
    x1_p = x1_p - stepper * x1_grad
    x2_p = x2_p - stepper * x2_grad
    loss = cal_loss(data_x1, data_x2, x1_p, x2_p)
    return x1_p, x2_p, loss


if __name__ == '__main__':
    for i in range(100000):
        index = np.random.randint(0, 1999)
        x1_r = x1[index]
        x2_r = x2[index]
        c1, c2, loss = stochastic_gradient_descent_step(c1, c2, x1_r, x2_r)
        print "Iter:", i, "Loss: ", loss
        if loss < 0.00001 or i >= 99999:
            print "Centroid are: ", c1, c2
            break
