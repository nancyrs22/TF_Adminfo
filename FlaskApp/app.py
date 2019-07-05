from flask import Flask, request
from flask import render_template
import pandas
import numpy as np
from Static.functions import *
from Static.Kmeans import *
from Static.KNN import *
import matplotlib.pyplot as plt
from math import pi
import os
from bokeh.plotting import figure, ColumnDataSource
from bokeh.embed import components

app = Flask(__name__)

exists = os.path.isfile('Static/dataset.csv')

if exists:
    dataset = pandas.read_csv("Static/dataset.csv")
else:
    dataset= pandas.read_csv("Static/weather_hourly_darksky.csv")

    #We only want to use the first 5000 tuples
    dataset = dataset.head(5000)

    #Normalize the data from numeric columns
    column = dataset['visibility']
    dataset['visibility'] = normalize(column)

    column = dataset['windBearing']
    dataset['windBearing'] = normalize(column)

    column = dataset['temperature']
    dataset['temperature'] = normalize(column)

    column = dataset['dewPoint']
    dataset['dewPoint'] = normalize(column)

    column = dataset['pressure']
    dataset['pressure'] = normalize(column)

    column = dataset['apparentTemperature']
    dataset['apparentTemperature'] = normalize(column)

    column = dataset['windSpeed']
    dataset['windSpeed'] = normalize(column)

    column = dataset['humidity']
    dataset['humidity'] = normalize(column)

    #Export values to a csv file
    dataset.to_csv('Static/dataset.csv', index=False)

labels = dataset['summary']
labels = labels.values.tolist()
dataset.drop(['time', 'precipType', 'icon', 'summary'], axis = 1, inplace=True)
columns = ['visibility','windBearing','temperature', 'dewPoint','pressure','apparentTemperature','windSpeed','humidity']
categories = ['Breezy', 'Breezy and Mostly Cloudy', 'Breezy and Overcast', 'Breezy and Partly Cloudy', 'Clear', 'Foggy',
'Mostly Cloudy','Overcast', 'Partly Cloudy','Windy and Partly Cloudy','Windy and Mostly Cloudy']

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/WA')
def WA():

    exists = os.path.isfile('Static/data/WADataset.csv')
    if exists:
        WADataset = pandas.read_csv('Static/data/WADataset.csv')
    else:
        WADataset = weighted_average(dataset,columns,[2,2,4,1,1,3,1,2])
        WADataset.to_csv('Static/data/WADataset.csv', index=False)

    exists = os.path.isfile('Static/images/radarWA.png')
    if not exists:
        radarAllPlot(WADataset.head(9), [0,1,2,3,4,5,6,7],labels, 'WA')

    return render_template('WA.html', url = 'static/images/radarWA.png')

@app.route('/MMin')
def MMin():

    exists = os.path.isfile('Static/data/MaximinDataset.csv')
    if exists:
        MaximinDataset = pandas.read_csv('Static/data/MaximinDataset.csv')
    else:
        MaximinDataset = maximin(dataset,columns)
        MaximinDataset.to_csv('Static/data/MaximinDataset.csv', index=False)

    exists = os.path.isfile('Static/images/radarMMin.png')
    if not exists:
        radarAllPlot(MaximinDataset.head(9), [0,1,2,3,4,5,6,7],labels, 'MMin')

    return render_template('MMin.html', url = 'static/images/radarMMin.png')

@app.route('/MMax')
def MMax():

    exists = os.path.isfile('Static/data/MMaxDataset.csv')

    if exists:
        MMaxDataset = pandas.read_csv('Static/data/MMaxDataset.csv')
    else:
        MMaxDataset = minimax(dataset,columns)
        MMaxDataset.to_csv('Static/data/MMaxDataset.csv', index=False)

    exists = os.path.isfile('Static/images/radarMMax.png')
    if not exists:
        radarAllPlot(MMaxDataset.head(9), [0,1,2,3,4,5,6,7],labels, 'MMax')

    return render_template('MMax.html', url = 'static/images/radarMMax.png')

@app.route('/LMin')
def LMin():

    exists = os.path.isfile('Static/data/LeximinDataset.csv')

    if exists:
        LeximinDataset = pandas.read_csv('Static/data/LeximinDataset.csv')
    else:
        LeximinDataset = leximin(dataset,columns)
        LeximinDataset.to_csv('Static/data/LeximinDataset.csv', index=False)

    exists = os.path.isfile('Static/images/radarLMin.png')
    if not exists:
        radarAllPlot(LeximinDataset.head(9), [0,1,2,3,4,5,6,7],labels, 'LMin')

    return render_template('LMin.html', url = 'static/images/radarLMin.png')

@app.route('/LMax')
def LMax():

    exists = os.path.isfile('Static/data/LeximaxDataset.csv')

    if exists:
        LeximaxDataset = pandas.read_csv('Static/data/LeximaxDataset.csv')
    else:
        LeximaxDataset = leximax(dataset,columns)
        LeximaxDataset.to_csv('Static/data/LeximaxDataset.csv', index=False)

    exists = os.path.isfile('Static/images/radarLMax.png')
    if not exists:
        radarAllPlot(LeximaxDataset.head(9), [0,1,2,3,4,5,6,7],labels, 'LMax')

    return render_template('LMax.html', url = 'static/images/radarLMax.png')

@app.route('/PCASkylines')
def Skylines():

    exists = os.path.isfile('Static/data/SkylinesDataset.csv')

    if exists:
        SkylinesDataset = pandas.read_csv('Static/data/SkylinesDataset.csv')
    else:
        data = pca(dataset,2)
        data = pandas.DataFrame(data, columns=["val1", "val2"])
        data['Summary'] = pandas.DataFrame(labels)
        SkylinesDataset = skylines(data,["val1", "val2"])
        SkylinesDataset.to_csv('Static/data/SkylinesDataset.csv', index=False)

    plot = dotPlot(SkylinesDataset, categories, "PCA + Skylines", "Summary")

	# Embed plot into HTML via Flask Render
    script, div = components(plot)
    return render_template("Skylines.html", script=script, div=div)

@app.route('/PCAKNN')
def PCAKNN():
    return render_template('index.html')

@app.route('/PCAKMEANS')
def PCAKMEANS():

    exists = os.path.isfile('Static/data/PCAKMEANSData.csv')
    cen = request.args.get("cen")

    if exists and cen == None:
        PCAKMEANSData = pandas.read_csv('Static/data/PCAKMEANSData.csv')
        centers = pandas.read_csv('Static/data/PCAKMEANSCenters.csv')
        centers = centers.values.tolist()
    else:
        if cen == None:
            cen = 11

        data = pca(dataset,2)
        data = pandas.DataFrame(data, columns=["val1", "val2"])
        data = data.values.tolist()
        centers, PCAKMEANSData = k_meansInit(data, int(cen), 10)
        PCAKMEANSData = pandas.DataFrame(PCAKMEANSData, columns=["val1", "val2","center"])
        PCAKMEANSData.to_csv('Static/data/PCAKMEANSData.csv', index=False)
        centers = pandas.DataFrame(centers)
        centers.to_csv('Static/data/PCAKMEANSCenters.csv',index = False)
        centers = centers.values.tolist()

    plot = dotPlot(PCAKMEANSData, [i for i in range(0,len(centers))], "PCA + K-means", "center")

    # Embed plot into HTML via Flask Render
    script, div = components(plot)
    return render_template("PCAKMEANS.html", script=script, div=div)

@app.route('/Histograms')
def Histograms():
    bins = request.args.get("bins")
    if bins == None:
        bins = 50
    else:
        bins = int(bins)

    # Create the plot
    plots = []

    for col in columns:

        plot = histogram(dataset, col, "Histogram of "+col,labels, bins)
        plots.append(components(plot))

    return render_template('Histograms.html',plots = plots)

@app.route('/<name>/Show')
def show(name):
    if name == 'WA':
        WADataset = pandas.read_csv('Static/data/WADataset.csv')
        return render_template('dataframe.html',  tables=[WADataset.to_html(classes='data', header = True)])
    if name == 'MMin':
        MaximinDataset = pandas.read_csv('Static/data/MaximinDataset.csv')
        return render_template('dataframe.html',  tables=[MaximinDataset.to_html(classes='data', header = True)])
    if name == 'MMax':
        MMaxDataset = pandas.read_csv('Static/data/MMaxDataset.csv')
        return render_template('dataframe.html',  tables=[MMaxDataset.to_html(classes='data', header = True)])
    if name == 'LMin':
        LeximinDataset = pandas.read_csv('Static/data/LeximinDataset.csv')
        return render_template('dataframe.html',  tables=[LeximinDataset.to_html(classes='data', header = True)])
    if name == 'LMax':
        LeximaxDataset = pandas.read_csv('Static/data/LeximaxDataset.csv')
        return render_template('dataframe.html',  tables=[LeximaxDataset.to_html(classes='data', header = True)])
    if name == 'Skylines':
        SkylinesDataset = pandas.read_csv('Static/data/SkylinesDataset.csv')
        return render_template('dataframe.html',  tables=[SkylinesDataset.to_html(classes='data', header = True)])
    if name == 'Pca+Kmeans':
        PCAKMEANSData = pandas.read_csv('Static/data/PCAKMEANSData.csv')
        return render_template('dataframe.html',  tables=[PCAKMEANSData.to_html(classes='data', header = True)])
    if name == 'Centers':
        PCAKMEANSCenters = pandas.read_csv('Static/data/PCAKMEANSCenters.csv')
        return render_template('dataframe.html',  tables=[PCAKMEANSCenters.to_html(classes='data', header = True)])

    return render_template(name+'.html')



############ PLOTS ###############

def radarPlot(df, row, categories, color,title, type):
    N = len(categories)

    valores = df.loc[df.index[row]].values[categories].flatten().tolist()
    valores += valores[:1]

    angulos = [n / float(N) * 2 * pi for n in range(N)]
    angulos += angulos[:1]

    ax = plt.subplot(3, 3, row + 1, polar=True, )
    plt.subplots_adjust(hspace = 0.3)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    etiquetas = [df.columns[i] for i in categories]
    plt.xticks(angulos[:-1], etiquetas, color='grey', size=8)
    ax.set_rlabel_position(0)

    tic = 5
    plt.yticks([round(i * (1.0 / tic),2) for i in range(1,tic)], [str(i * (1.0 / tic)) for i in range(1,tic)], color="grey", size=7)
    plt.ylim(0,1)

    ax.plot(angulos, valores, color=color, linewidth=2, linestyle='solid')
    ax.fill(angulos, valores, color=color, alpha=0.4)
    plt.title(title, size=11, color=color, y=1.1)
    plt.savefig('Static/images/radar' + type + '.png')



def radarAllPlot(df,categories, labels, type):

    my_dpi=96
    plt.figure(figsize=(1500/my_dpi, 1500/my_dpi), dpi=my_dpi)
    my_palette = plt.cm.get_cmap("Set2", len(df.index))
    for i in range(len(df.index)):
        radarPlot(df,i,categories,my_palette(i), labels[i], type)



def dotPlot(df, categories, title, label):

    colormap = {}
    c = ['blue','brown','chartreuse','darkblue','darkgreen','gold','magenta','slategrey',
    'yellow','lightskyblue', 'black']

    for i in range(0,len(categories)):
        colormap[categories[i]] = c[i]

    colors = [colormap[x] for x in df[label]]

    source = ColumnDataSource(data = dict(
        Val1 = df["val1"].values.tolist(),
        Val2 = df["val2"].values.tolist(),
        summary = df[label].values.tolist(),
        color=colors,
    ))

    TOOLTIPS=[
        ("Val1", "@Val1"),
        ("Val2", "@Val2"),
        ("label", "@summary")
    ]

    p = figure(plot_width=1500, plot_height=700, title = title, tooltips = TOOLTIPS)
    p.xaxis.axis_label = 'Val1'
    p.yaxis.axis_label = 'Val2'

    p.circle(x='Val1', y='Val2', color = 'color', fill_alpha=0.2, size=10, source=source)

    return p

def histogram(df, label, title, labels, bins):
    data = df[label].values.tolist()
    hist, edges = np.histogram(data, bins=bins,density=True)

    p = figure(plot_width=1500, plot_height=700,title=title, background_fill_color="#fafafa")

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)

    p.xaxis.axis_label = label
    p.yaxis.axis_label = 'count'
    p.grid.grid_line_color="white"
    return p

# With debug=True, Flask server will auto-reload
# when there are code changes
if __name__ == '__main__':
	app.run(port=5000, debug=True)
