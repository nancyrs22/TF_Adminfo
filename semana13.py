K Nearest Neighbor (K vecinos más cercano)
1. Teoria
1.1 Algoritmo
KNN se puede utilizar para problemas tanto de clasificación como de predicción. KNN cae en la familia de algoritmos de aprendizaje supervisado. De manera informal, esto significa que se nos da un conjunto de datos etiquetado que contiene observaciones de capacitación  (x,y)  y nos gustaría capturar la relación entre  x  y  y . Más formalmente, nuestro objetivo es aprender una función  h:X→Y  para que, dada una observación nueva  x ,  h(x)  pueda predecir con confianza la salida correspondiente  y .

1.2 Metrica de distancia
En la configuración de clasificación, el algoritmo KNN se reduce esencialmente a formar un voto mayoritario entre las K instancias más similares a una observación "no vista" dada. La similitud se define de acuerdo con una métrica de distancia entre dos puntos de datos. El clasificador KNN se basa comúnmente en la distancia euclidiana entre una muestra de prueba y las muestras de entrenamiento especificadas. Sea  xi  una muestra de entrada con  p features  (xi1,xi2,...,xip) ,  n  sea el número total de muestras de entrada  (i=1,2,...,n) . La distancia euclidiana entre la muestra  xi  y  xl  se define como:

d(xi,xl)=(xi1−xl1)2+(xi2−xl2)2+...+(xip−xlp)2−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−√

A veces, otras medidas pueden ser más adecuadas para una configuración determinada e incluyen la distancia de Manhattan, Chebyshev y Hamming.

1.3 Pasos del algoritmo
PASO 1: Elegir el número K de vecinos.

PASO 2: Tome los K vecinos más cercanos del nuevo punto de datos, de acuerdo con su métrica de distancia

PASO 3: Entre estos K vecinos, cuente la cantidad de puntos de datos para cada categoría

PASO 4: Asigne el nuevo punto de datos a la categoría en la que contó más vecinos

2. Importación y preparación de datos.
2.1 Importación de librerias
import numpy as np
import pandas as pd
2.2 Cargar el dataset
NOTA: el conjunto de datos del iris incluye tres especies de iris con 50 muestras cada una, así como algunas propiedades de cada flor. Una especie de flor es linealmente separable de las otras dos, pero las otras dos no son linealmente separables una de la otra.

dataset = pd.read_csv('../input/Iris.csv')
2.3 Resumir el conjunto de datos
# Podemos tener una idea rápida de cuántas instancias (filas) y cuántos atributos (columnas) contienen los datos con la propiedad de forma.
dataset.shape
(150, 6)
dataset.head(5)

dataset.describe()

# Veamos ahora la cantidad de instancias (filas) que pertenecen a cada clase. Podemos ver esto como una cuenta absoluta.
dataset.groupby('Species').size()

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = dataset[feature_columns].values
y = dataset['Species'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from pandas.plotting import parallel_coordinates
plt.figure(figsize=(15,10))
parallel_coordinates(dataset.drop("Id", axis=1), "Species")
plt.title('Coordenadas Paralelas', fontsize=20, fontweight='bold')
plt.xlabel('Caracteristicas', fontsize=15)
plt.ylabel('Valores de las Caracteristicas', fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()

from pandas.plotting import andrews_curves
plt.figure(figsize=(15,10))
andrews_curves(dataset.drop("Id", axis=1), "Species")
plt.title('Andrews Curves Plot', fontsize=20, fontweight='bold')
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()

plt.figure()
sns.pairplot(dataset.drop("Id", axis=1), hue = "Species", size=3, markers=["o", "s", "D"])
plt.show()

plt.figure()
dataset.drop("Id", axis=1).boxplot(by="Species", figsize=(15, 10))
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(20, 15))
ax = Axes3D(fig, elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s = X[:, 3]*50)

for name, label in [('Virginica', 0), ('Setosa', 1), ('Versicolour', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean(),
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),size=25)

ax.set_title("3D visualization", fontsize=40)
ax.set_xlabel("Sepal Length [cm]", fontsize=25)
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Sepal Width [cm]", fontsize=25)
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Petal Length [cm]", fontsize=25)
ax.w_zaxis.set_ticklabels([])

plt.show()
