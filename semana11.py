#KMEANS 1
# inicializaciones
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 8) # (w, h)
from sklearn import metrics
from sklearn.cluster import KMeans

# Datos: leyendo un archivo CSV (creado por 00-gen2Ddata.ipynb)
D = pd.read_csv('kms-dataset2d-X.csv', sep=';', header=None)
print(D.describe())
X = np.array(D)

# Muestra los datos
# Plot
plt.scatter(X[:,0], X[:,1])
plt.show()

# k-means:
kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
kmeans.fit(X)

# Calcula los valores en los puntos de una cuadrícula 2D (malla)
h = .1     # punto en la malla [x_min, x_max] x [y_min, y_max].

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# Generar una cuadrícula de pasos.
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# etiquetas de puntos de grilla
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
print(Z.shape)

# Muestra
plt.figure(1)
plt.clf()
# Mostrar los puntos de la cuadrícula, como una imagen (interpolación) que da el color de fondo.
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
# Añade los puntos
plt.plot(X[:, 0], X[:, 1], 'b.', markersize=5)
# Muestra los centroides
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means bidimensionnel')
plt.show()

#KMEASN 2
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 8) # (w, h)
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.datasets import load_sample_image
import imageio
from time import time

img = imageio.imread('Cusco.jpg')

plt.axis('off')
plt.title('Imagen Original')
plt.imshow(img)
plt.show()

print(type(img))
print(img.shape)
print(img[0,10]) # el décimo píxel de la línea 0

img = np.array(img, dtype=np.float64) / 255

w, h, d = img.shape
assert d == 3 # solo procesar imagen en color RGB
image_array = img.reshape(w * h, d) # Una matriz de píxeles, sin una estructura 2D.
print(image_array.shape)

A = np.array( [ (0,0,0), (1,1,1), (1,2,3), (0,0,0)])
print(A)
print('Unico: \n', np.unique(A, axis=0))

print('Hay', len(np.unique(image_array, axis=0)), 'colores distintos sobre', w*h, 'pixels' )

n_colors = 16 # N
codebook_random = shuffle(image_array, random_state=0)[:n_colors]

t0 = time()
labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
print("hecho en %0.3fs." % (time() - t0))

plt.imshow(labels_random.reshape(w, h))
plt.title('Píxeles de colores similares (aleatorio,% d colores)' % n_colors)
plt.axis('off')

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

img_q_random = recreate_image(codebook_random, labels_random, w, h)

plt.axis('off')
plt.title('Imagen cuantificada sobre %d colores aleatorios' % n_colors)
plt.imshow(img_q_random)
plt.show()

print("Ajuste del modelo en una pequeña sub-muestra de los datos.")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("hecho en %0.3fs." % (time() - t0))

print("Predicción de índices de color en la imagen completa (k-medias)")
t0 = time()
labels_kmeans = kmeans.predict(image_array)
print("hecho en %0.3fs." % (time() - t0))

plt.imshow(labels_kmeans.reshape(w, h))
plt.title('Pixels de colores similares (k-means, %d colores)' % n_colors)
plt.axis('off')

Imagen cuantificada construida.
img_q_kmeans = recreate_image(kmeans.cluster_centers_, labels_kmeans, w, h)
Mostrar el resultado
plt.axis('off')
plt.title('Imagen cuantificada sobre %d colores k-means' % n_colors)
plt.imshow(img_q_kmeans)
plt.show()

Mostrando tres imágenes lado a lado
plt.figure(figsize=(20,4))

plt.subplot(1, 3, 1)
plt.axis('off')
plt.title('Imagen original (96615 colores)')
plt.imshow(img)

plt.subplot(1, 3, 2)
plt.axis('off')
plt.title('Cuantificada sobre %d colores k-means' % n_colors)
plt.imshow(recreate_image(kmeans.cluster_centers_, labels_kmeans, w, h))

plt.subplot(1, 3, 3)
plt.axis('off')
plt.title('Cuantificada sobre %d colores aleatorios' % n_colors)
plt.imshow(recreate_image(codebook_random, labels_random, w, h))

plt.show()
