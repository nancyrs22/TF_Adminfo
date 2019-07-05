import pandas
import matplotlib.pyplot  as plt
import numpy as np
from functions import normalize

#Read the dataset to start working with the data
dataset= pandas.read_csv("weather_hourly_darksky.csv")

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
dataset.to_csv("dataset.csv", index=False)
print(dataset)