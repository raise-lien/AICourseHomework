import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
y_train = train_data['Survived']

random_forest = RandomForestClassifier(n_estimators=250, max_depth=5, criterion='gini')
random_forest.fit(x_train, y_train)
Y_pred = random_forest.predict(x_test)

submission = pd.DataFrame({'PassengerId': test_data['PassengerId'],
                           'Survived': Y_pred})
submission.to_csv("titanic_rf_submission.csv", index=False)