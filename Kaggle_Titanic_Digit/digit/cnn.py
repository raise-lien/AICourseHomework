import pandas as pd
import tensorflow as tf
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv").values

images = train.iloc[:, 1:].values
labels_flat = train.iloc[:, 0].values.ravel()

images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)

label_count = np.unique(labels_flat).shape[0]


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


labels = dense_to_one_hot(labels_flat, 10)
labels = labels.astype(np.uint8)

validation_size = 2000
validation_images = images[:validation_size]
validation_labels = labels[:validation_size]

train_images = images[validation_size:]
train_labels = labels[validation_size:]

batch_size = 100
n_batch = len(train_images) / batch_size

x = tf.placeholder('float', shape=[None, 784])
y = tf.placeholder('float', shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")




x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([6, 6, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(150):
        total_loss = 0
        for batch in range(n_batch):
            batch_x = train_images[batch * batch_size:(batch + 1) * batch_size]
            batch_y = train_labels[batch * batch_size:(batch + 1) * batch_size]
            _, e_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            total_loss += e_loss
        epoch_accuracy = sess.run(accuracy, feed_dict={x: validation_images, y: validation_labels, keep_prob: 1.0})
        print("第" + str(epoch + 1) + "轮，准确率为：" + str(epoch_accuracy) + " Loss为：" + str(e_loss))

    test_x = np.array(test, dtype=np.float32)
    y_predict = y_conv.eval(feed_dict={x: test_x[1:100, :], keep_prob: 1.0})
    y_predict_all = list()
    for i in np.arange(100, 28001, 100):
        y_predict = y_conv.eval(feed_dict={x: test_x[i - 100:i, :], keep_prob: 1.0})
        test_pred = np.argmax(y_predict, axis=1)
        y_predict_all = np.append(y_predict_all, test_pred)
    submission = pd.DataFrame({"ImageID": range(1, 28001), "Label": np.int32(y_predict_all)})
    submission.to_csv("cnn_submission.csv", index=False)
