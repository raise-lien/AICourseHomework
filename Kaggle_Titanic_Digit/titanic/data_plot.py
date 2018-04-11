# coding=utf-8
import pandas as pd
import seaborn as sns

train_data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

print train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                                ascending=False)
print train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',
                                                                                          ascending=False)
print train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False)
print train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False)
print train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                    ascending=False)
train_data['Title'] = train_data.Name.str.extract('([A-Za-z]+)\.', expand=False)
train_data['Title'] = train_data['Title'].replace(
    ['Lady', 'Countess', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
train_data['Title'] = train_data['Title'].map(title_map)
train_data['Title'] = train_data['Title'].fillna(0)

print train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived',
                                                                                                    ascending=False)
grid = sns.FacetGrid(train_data, 'Survived')
grid.map(plt.hist, 'Age', bins=20)
sns.set_style('darkgrid')
fig1 = plt.figure(1)


grid = sns.FacetGrid(train_data, row='Pclass', col='Survived', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
sns.set_style('darkgrid')
fig2 = plt.figure(2)

grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)
sns.set_style('darkgrid')
fig3 = plt.figure(3)


Survived_cabin = train_data.Survived[pd.notnull(train_data.Cabin)].value_counts()
Survived_nocabin = train_data.Survived[pd.isnull(train_data.Cabin)].value_counts()
df = pd.DataFrame({u'Yes': Survived_cabin, u'No': Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.xlabel(u"Cabin or not")
plt.ylabel(u"People")
fig4 = plt.figure(4)
fig4.set(alpha=0.2)


plt.show()
