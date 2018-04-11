import numpy as np
import pandas as pd
import tensorflow as tf

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print train_data.info()
print test_data.info()

print train_data['Embarked'].value_counts()

print test_data['Embarked'].value_counts()

train_data['Title'] = train_data.Name.str.extract('([A-Za-z]+)\.', expand=False)
train_data['Title'] = train_data['Title'].replace(
    ['Lady', 'Countess', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
train_data['Title'] = train_data['Title'].map(title_map)
train_data['Title'] = train_data['Title'].fillna(0)

test_data['Title'] = test_data.Name.str.extract('([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Title'].replace(
    ['Lady', 'Countess', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
test_data['Title'] = test_data['Title'].map(title_map)
test_data['Title'] = test_data['Title'].fillna(0)

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0
train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1
test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0
test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1

train_data['Embarked'].fillna('S', inplace=True)
train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2
test_data.loc[test_data['Embarked'] == 'S', 'Embarked'] = 0
test_data.loc[test_data['Embarked'] == 'C', 'Embarked'] = 1
test_data.loc[test_data['Embarked'] == 'Q', 'Embarked'] = 2

print train_data.info()
print test_data.info()

x_train = train_data[['Title', 'Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']].as_matrix()
x_test = test_data[['Title', 'Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']].as_matrix()
train_data['Dead'] = train_data['Survived'].apply(lambda s: 1 - s)
y_train = train_data[['Dead', 'Survived']].as_matrix()


X = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 2])
weights = tf.Variable(tf.random_normal([8, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')
y_pred = tf.nn.softmax(tf.matmul(X, weights) + bias)
cross_entropy = - tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(150):
        total_loss = 0.
        for i in range(len(x_train)):
            feed_dict = {X: [x_train[i]], y: [y_train[i]]}
            _, loss = sess.run([train_op, cost], feed_dict=feed_dict)
            total_loss += loss
        print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))
    print("Train Complete")
    predictions = np.argmax(sess.run(y_pred, feed_dict={X: x_test}), axis=1)
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("titanic-submission_new.csv", index=False)
