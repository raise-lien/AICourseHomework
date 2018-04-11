import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.uniform(low=0.0, high=1.0, size=2000)
x2 = np.random.uniform(low=0.0, high=1.0, size=2000)

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_title('2000 Points Plot')
# plt.xlabel('X1')
# plt.ylabel('X2')
# ax1.scatter(x1, x2, s=10)
# plt.show()


c1 = 0.0
c2 = 0.0


def cal_loss(x1_i, x2_i, x1_c, x2_c):
    return (x1_i - x1_c)**2 + (x2_i - x2_c)**2


def cal_pra_c1(x1_i, x1_c):
    return 2 * (x1_c - x1_i)


def cal_pra_c2(x2_i, x2_c):
    return 2 * (x2_c - x2_i)


def gradient_descent_step(x1_p, x2_p, x1, x2):
    n_points = 2000
    stepper = 0.01
    loss = 0.0
    x1_grad = 0
    x2_grad = 0
    for i in range(n_points):
        data_x1 = x1[i]
        data_x2 = x2[i]
        x1_grad += cal_pra_c1(data_x1, x1_p) / 2
        x2_grad += cal_pra_c2(data_x2, x2_p) / 2
        # print x1_grad, x2_grad
    x1_p = x1_p - stepper * (x1_grad / n_points)
    x2_p = x2_p - stepper * (x2_grad / n_points)
    for j in range(n_points):
        data_x1 = x1[j]
        data_x2 = x2[j]
        loss += cal_loss(data_x1, data_x2, x1_p, x2_p)
    loss = loss / (2 * n_points)
    return x1_p, x2_p, loss


if __name__ == '__main__':
    for i in range(1000):
        c1, c2, loss = gradient_descent_step(c1, c2, x1, x2)
        print "Iter:", i, "Loss: ", loss
        if loss < 0.01 or i >= 999:
            print "Centroid are: ", c1, c2
            break
