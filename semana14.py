import pandas as pd
#iris = pd.read_csv('iris.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
iris = pd.read_csv('iris.csv')
print(iris.head())

wine_reviews = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)
wine_reviews.head()

import matplotlib.pyplot as plt

# create a figure and axis
fig, ax = plt.subplots()

# scatter the sepal_length against the sepal_width
ax.scatter(iris['sepallength'], iris['sepalwidth'])
# set a title and labels
ax.set_title('Iris Dataset')
ax.set_xlabel('sepallength')
ax.set_ylabel('sepalwidth')

#create color dictionary
colors = {'Iris-setosa':'r', 'Iris-versicolor':'g', 'Iris-virginica':'b'}
# create a figure and axis
fig, ax = plt.subplots()
# plot each data-point
for i in range(len(iris['sepallength'])):
    ax.scatter(iris['sepallength'][i], iris['sepalwidth'][i],color=colors[iris['class'][i]])
# set a title and labels
ax.set_title('Iris Dataset')
ax.set_xlabel('sepallength')
ax.set_ylabel('sepalwidth')

# get columns to plot
columns = iris.columns.drop(['class'])
# create x data
x_data = range(0, iris.shape[0])
# create figure and axis
fig, ax = plt.subplots()
# plot each column
for column in columns:
    ax.plot(x_data, iris[column])
# set title and legend
ax.set_title('Iris Dataset')
ax.legend()

# create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(wine_reviews['points'])
# set title and labels
ax.set_title('Wine Review Scores')
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')

# create a figure and axis
fig, ax = plt.subplots()
# count the occurrence of each class
data = wine_reviews['points'].value_counts()
# get x and y data
points = data.index
frequency = data.values
# create bar chart
ax.bar(points, frequency)
# set title and labels
ax.set_title('Wine Review Scores')
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')

iris.plot.scatter(x='sepallength', y='sepalwidth', title='Iris Dataset')
iris.drop(['class'], axis=1).plot.line(title='Iris Dataset')
wine_reviews['points'].plot.hist()
iris.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)
wine_reviews['points'].value_counts().sort_index().plot.bar()
wine_reviews['points'].value_counts().sort_index().plot.barh()
wine_reviews.groupby("country").price.mean().sort_values(ascending=False)[:5].plot.bar()

import seaborn as sns
sns.scatterplot(x='sepallength', y='sepalwidth', data=iris)

sns.lineplot(data=iris.drop(['class'], axis=1))
sns.distplot(wine_reviews['points'], bins=10, kde=False)
sns.distplot(wine_reviews['points'], bins=10, kde=True)
sns.countplot(wine_reviews['points'])

df = wine_reviews[(wine_reviews['points']>=95) & (wine_reviews['price']<1000)]
sns.boxplot('points', 'price', data=df)

import numpy as np
# get correlation matrix
corr = iris.corr()
fig, ax = plt.subplots()
# create heatmap
im = ax.imshow(corr.values)

# set labels
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# get correlation matrix
corr = iris.corr()
fig, ax = plt.subplots()
# create heatmap
im = ax.imshow(corr.values)

# set labels
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        text = ax.text(j, i, np.around(corr.iloc[i, j], decimals=2),
                       ha="center", va="center", color="black")

sns.heatmap(iris.corr(), annot=True)

g = sns.FacetGrid(iris, col='class')
g = g.map(sns.kdeplot, 'sepallength')

sns.pairplot(iris)
from pandas.plotting import scatter_matrix

fig, ax = plt.subplots(figsize=(12,12))
scatter_matrix(iris, alpha=1, ax=ax)
