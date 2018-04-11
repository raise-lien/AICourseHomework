# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv").values
images = train.iloc[:, 1:].values
labels_flat = train.iloc[:, 0].values.ravel()

images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
label_count = np.unique(labels_flat).shape[0]
labels = dense_to_one_hot(labels_flat, 10)
labels = labels.astype(np.uint8)

validation_size = 2000
validation_images = images[:validation_size]
validation_labels = labels[:validation_size]
train_images = images[validation_size:]
train_labels = labels[validation_size:]



x = tf.placeholder('float', shape=[None, 784])
y = tf.placeholder('float', shape=[None, 10])

weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, weights)+biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(prediction, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 100
n_batch = len(train_images)/batch_size
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(150):
        total_loss = 0
        for batch in range(n_batch):
            batch_x = train_images[batch*batch_size:(batch+1)*batch_size]
            batch_y = train_labels[batch*batch_size:(batch+1)*batch_size]
            _, e_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
            total_loss += e_loss
        epoch_accuracy = sess.run(accuracy, feed_dict={x: validation_images, y: validation_labels})
        print("第"+str(epoch+1)+"轮，准确率为："+str(epoch_accuracy)+" Loss为："+str(total_loss))

    test_x = np.array(test, dtype=np.float32)
    y_predict = prediction.eval(feed_dict={x: test_x[1:100, :]})
    y_predict_all = list()
    for i in np.arange(100, 28001, 100):
        y_predict = prediction.eval(feed_dict={x: test_x[i-100:i, :]})
        test_pred = np.argmax(y_predict, axis=1)
        y_predict_all = np.append(y_predict_all, test_pred)
    submission = pd.DataFrame({"ImageID": range(1, 28001), "Label": np.int32(y_predict_all)})
    submission.to_csv("linear_submission.csv", index=False)