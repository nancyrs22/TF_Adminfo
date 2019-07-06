%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');

A simple vista, está claro que existe una relación casi lineal entre las variables x e y.

En el análisis de componentes principales, esta relación se cuantifica encontrando una lista de los ejes principales en los datos, y utilizando esos ejes para describir el conjunto de datos. Usando el estimador PCA de Scikit-Learn, podemos calcular esto de la siguiente manera:

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
El entrenamiento aprende algunas cantidades de los datos, lo que es más importante, los "componentes" y la "varianza explicada":

print(pca.components_)
[[-0.94446029 -0.32862557]
 [-0.32862557  0.94446029]]
print(pca.explained_variance_)
[0.7625315 0.0184779]
Para ver qué significan estos números, visualicemos como vectores sobre los datos de entrada, usando los "componentes" para definir la dirección del vector, y la "varianza explicada" para definir la longitud al cuadrado del vector:

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)
original shape:    (200, 2)
transformed shape: (200, 1)
Los datos transformados se han reducido a una sola dimensión. Para comprender el efecto de esta reducción de la dimensionalidad, podemos realizar la transformación inversa de estos datos reducidos y trazarla junto con los datos originales:

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

pca = PCA(2)  # proyectar de 64 a 2 dimensiones
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('componente 1')
plt.ylabel('componente 2')
plt.colorbar();

pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('numero de componentes')
plt.ylabel('varianza explicada acumulada');

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
plot_digits(digits.data)

np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)

import numpy as np
import matplotlib.pyplot as plt
Ejemplo: Eigenfaces
Aquí exploramos un ejemplo del uso de una proyección de PCA como selector de características para el reconocimiento facial con una máquina de vectores de soporte. Aquí vamos a echar un vistazo atrás y explorar un poco más de lo que pasó en eso. Recuerde que estábamos utilizando las Caras etiquetadas en el conjunto de datos Wild disponible a través de Scikit-Learn:

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush'
 'Gerhard Schroeder' 'Hugo Chavez' 'Junichiro Koizumi' 'Tony Blair']
(1348, 62, 47)
Echemos un vistazo a los ejes principales que abarcan este conjunto de datos. Debido a que este es un conjunto de datos grande, usaremos el estimador estándar de PCA útil para Datos tridimensionales (aquí, una dimensionalidad de casi 3,000). Vamos a echar un vistazo a los primeros 150 componentes:

from sklearn.decomposition import PCA
pca = PCA(150)
pca.fit(faces.data)
PCA(copy=True, iterated_power='auto', n_components=150, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
En este caso, puede ser interesante visualizar las imágenes asociadas con los primeros componentes principales (estos componentes se conocen técnicamente como "vectores propios") por lo que estos tipos de imágenes a menudo se denominan "interfaces propias"). Como puedes ver en esta figura, son tan espeluznantes como suenan:

fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')

    Los resultados son muy interesantes y nos dan una idea de cómo varían las imágenes: por ejemplo, las primeras interfaces propias (desde la parte superior izquierda) parecen estar asociadas con el ángulo de iluminación de la cara, y luego los vectores principales parecen estar recogiendo Por ciertos rasgos, como ojos, narices y labios. Echemos un vistazo a la varianza acumulada de estos componentes para ver cuánta información de datos conserva la proyección:

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('numero de componentes')
plt.ylabel('varianza explicada acumulada');

# Calcular los PCA y caras proyectadas
pca = PCA(150).fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)
# Plot de los resultados
fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')

ax[0, 0].set_ylabel('total-dim\noriginal')
ax[1, 0].set_ylabel('150-dim\nreconstruccion');
