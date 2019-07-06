import numpy as np
import pandas as pd

#cargar el dataset
data = pd.read_csv("videogames.csv")

#forma del dataset 1770 tuplas con 5 columnas
data.shape
#5 primeras filas
data.head()
def normalizacionColumna(df, i):
    columns = df.columns.values
    df[columns[i]] = (df[columns[i]] - data[columns[i]].min()) / (df[columns[i]].max() - data[columns[i]].min())

    def normalizarDataset(data, indices):
    df = data.copy()
    for i in indices:
        normalizacionColumna(df, i)
    return df

k = 10
indices = [2,3,4]
data = normalizarDataset(data, indices)
data.head(k)

def weightedAverage(data, w, l):
    df = data.copy()
    columns = df.columns.values
    wa = np.zeros(df.shape[0])
    for i in range(len(l)):
        wa += (w[i] * df[columns[l[i]]]) / sum(w)
    df['wa'] = wa
    df = df.sort_values(by=['wa'], ascending=False)
    return df

    pesos = [1,2,1]
dfWA = weightedAverage(data, pesos, indices)
dfWA.head(k)

def maximin(data, l):
    df = data.copy()
    columns = df.columns.values
    t = df.shape[0]
    mn = np.zeros(df.shape[0])
    for i in range(t):
        mn[i] = df[columns[l[0]]][i]
        for j in range(1,len(l)):
            if mn[i] > df[columns[l[j]]][i]:
                mn[i] = df[columns[l[j]]][i]
    df['minVal'] = mn
    #print(mn)
    df = df.sort_values(by=['minVal'], ascending=False)
    return df

    dfMM = maximin(data, indices)
dfMM.head(k)

def leximin(data, l):
    df = data.copy()
    columns = df.columns.values
    t = df.shape[0]
    lex = [np.zeros(df.shape[0]) for i in range(len(l))]
    a = [[]  for i in range(len(l))]
    for i in range(t):
        for j in range(len(l)):
            a[j] = df[columns[l[j]]][i]
        a.sort()
        for j in range(len(l)):
            lex[j][i] = a[j]
    for j in range(len(l)):
        df['c' + str(j)] = lex[j]
    c = ['c' + str(i) for i in range(len(l))]
    df = df.sort_values(by=c, ascending=False)
    return df

    dfLM = leximin(data, indices)
dfLM.head(k)

def ParetoDomina(a,b):
    mi = len([1 for i in range(len(a)) if a[i] >= b[i]])
    my = len([1 for i in range(len(a)) if a[i] > b[i]])
    if mi == len(a):
        if my > 0:
            return True
    return False

def skylines(data, l):
    df = data.copy()
    columns = df.columns.values
    t = df.shape[0]
    for i in range(t):
        if i in df.index:
            a = [0] * len(l)
            for j in range(i + 1, t):
                if j in df.index:
                    b = [0] * len(l)
                    for k in range(len(l)):
                        a[k] = df[columns[l[k]]][i]
                        b[k] = df[columns[l[k]]][j]
                    if ParetoDomina(a,b):
                        df = df.drop(j)
                    elif ParetoDomina(b,a):
                        df = df.drop(i)
                        break
    return df

dfSky = skylines(data, indices)
dfSky

import matplotlib.pyplot as plt
from math import pi

def radarPlot(df, row, categorias, color,title):
    N = len(categorias)
    #repetir el primer valor para tener una figura cerrada (poligono)
    valores = df.loc[df.index[row]].values[categorias].flatten().tolist()
    valores += valores[:1]
    #calcular el angulo
    angulos = [n / float(N) * 2 * pi for n in range(N)]
    angulos += angulos[:1]
    #inicializar el plot
    ax = plt.subplot(3, 2, row + 1, polar=True, )
    # primer eje arriba:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    #un eje por atributo + etiquetas
    etiquetas = [df.columns[i] for i in categorias]
    plt.xticks(angulos[:-1], etiquetas, color='grey', size=8)
    ax.set_rlabel_position(0)
    #dibujar ticks de los ejes
    tic = 5
    plt.yticks([i * (1.0 / tic) for i in range(1,tic)], [str(i * (1.0 / tic)) for i in range(1,tic)], color="grey", size=7)
    plt.ylim(0,1)
    #plotear
    ax.plot(angulos, valores, color=color, linewidth=2, linestyle='solid')
    ax.fill(angulos, valores, color=color, alpha=0.4)
    plt.title(title, size=11, color=color, y=1.1)

def radarAllPlot(df,categorias):
    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))
    for i in range(len(df.index)):
        #print(df.columns[1])
        #print()
        radarPlot(df,i,categorias,my_palette(i), df[df.columns[1]][df.index[i]])
        #radarPlot(df,i,categorias,my_palette(i), df["Title"][i])

    #4 primeros juegos segun WA
radarAllPlot(dfWA.head(4),indices)

#4 primeros juegos segun MM
radarAllPlot(dfMM.head(4),indices)

#4 primeros juegos segun MM
radarAllPlot(dfLM.head(4),indices)

#los 6 skylines juegos
radarAllPlot(dfSky.head(6),indices)
