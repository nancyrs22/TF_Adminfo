import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data = pd.read_csv("videogames.csv")
data.head()
# Checking if any values missing
data.isna().sum()
data[data['Weight'].isna()].head()
data.dropna(subset = ['Weight','Height'],inplace = True)
data.isna().sum()

## Converting players market value into value in Millions(M)
data['Value'] = data['Value'].fillna('NaN')
data['Value'] = data['Value'].apply(lambda x:
                                    float(re.findall('€(.*)M',x)[0]) if 'M' in x
                                    else (float(re.findall('€(.*)K',x)[0])/1000 if 'K' in x  else 0))

## Converting players wages into value in Thousands (K)
data['Wage'] = data['Wage'].fillna('NaN')
data['Wage'] = data['Wage'].apply(lambda x:float(re.findall('€(.*)K',x)[0]) if 'K' in x
                                  else float(re.findall('€(.*)',x)[0])/1000)

## Converting players release clause in Millions (M)
data['Release Clause'] = data['Release Clause'].fillna('NaN')
data['Release Clause'] = data['Release Clause'].apply(lambda x:
                                    float(re.findall('€(.*)M',x)[0]) if 'M' in x
                                    else (float(re.findall('€(.*)K',x)[0])/1000 if 'K' in x  else 0))

print("Total players in fifa 19 - ",data.shape[0])

tm = data['Preferred Foot'].value_counts()
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=500,
    height=500,
    title = "Count of players prefered foot"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

print("Total number of positions in FIFA19 is",data['Position'].nunique())
tm = data['Position'].value_counts()
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    title = "Count of the Players Playing in a particular Position"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

tm = data['Position'].value_counts(normalize=True)*100
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    title = "Percentage of Players Playing in a particular Position"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

forwards = ['ST','LF','RF','CF','LW','RW']
midfielders = ['CM','LCM','RCM','RM','LM','CDM','LDM','RDM','CAM','LAM','RAM','LCM','RCM']
defenders = ['CB','RB','LB','RCB','LCB','RWB','LWB']
goalkeepers = ['GK']
data['Overall_position'] = None
forward_players = data[data['Position'].isin(forwards)]
midfielder_players = data[data['Position'].isin(midfielders)]
defender_players = data[data['Position'].isin(defenders)]
goalkeeper_players = data[data['Position'].isin(goalkeepers)]
data.loc[forward_players.index,'Overall_position'] = 'forward'
data.loc[defender_players.index,'Overall_position'] = 'defender'
data.loc[midfielder_players.index,'Overall_position'] = 'midfielder'
data.loc[goalkeeper_players.index,'Overall_position'] = 'goalkeeper'

# sns.countplot(data['Overall_position'])
tm = data['Overall_position'].value_counts()
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=500,
    height=500,
    title = "Total players playing in the Overall position"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

print("TOP 5 FORWARDS")
data[data['Overall_position'] == 'forward'].sort_values(by = 'Overall', ascending = False).head()

print("TOP 5 MIDFIELDERS")
data[data['Overall_position'] == 'midfielder'].sort_values(by = 'Overall', ascending = False).head()

print("TOP 5 DEFENDERS")
data[data['Overall_position'] == 'defender'].sort_values(by = 'Overall', ascending = False).head()

print("TOP 5 GOALKEEPERS")
data[data['Overall_position'] == 'goalkeeper'].sort_values(by = 'Overall', ascending = False).head()

print("--------------Top 10 Highest Market Value in Millions € -------------- ")
data.sort_values(by = 'Value',ascending = False)[['Name','Age','Value','Overall','Potential','Position']].head(10)

print("--------------Top 10 Highest Wages Earned in Thousands € -------------- ")
data.sort_values(by = 'Wage',ascending = False)[['Name','Age','Wage','Overall','Potential','Position']].head(10)

print("--------------Top 10 Highest Release Clause in Millions €-------------- ")
data.sort_values(by = 'Release Clause',ascending = False)[['Name','Age','Release Clause','Overall','Potential','Position']].head(10)

tm = data.groupby('Nationality').count()['ID'].sort_values(ascending = False)
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=5000,
    height=600,
    title = "Total players from a Nation in the whole game"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

tm = data.groupby('Nationality').mean()['Overall'].sort_values(ascending = False)
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=True,
    width=5000,
    height=500,
    title = "Average overall rating of a player from a Nation in the whole game"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

## Creating national team data using group by
national_team_data = data.groupby(['Nationality'],as_index = False).agg(['mean','count','sum'])

national_team_data.head()

## So we consider average overall for those team which have atleast 200 players
tm = national_team_data[national_team_data['ID']['count']>200]['Overall']['mean'].sort_values(ascending = False)
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=1000,
    height=500,
    title = "Average Overall rating of a player from a Nation in the whole game (having atleast 200 players)"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

tm = national_team_data['Value']['sum'].sort_values(ascending = False).head(50)
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=1500,
    height=500,
    title = "Total valuation of players of a Nation in the whole game in Million(M) €"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)


tm = national_team_data[national_team_data['ID']['count']>100]['Value']['mean'].sort_values(ascending = False)
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=1000,
    height=500,
    title = "Average valuation of a player of a Nation in the whole game in Million(M) € (Nation having >100 players)"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

tm = national_team_data[national_team_data['ID']['count']>100]['Wage']['sum'].sort_values(ascending = False)
plt_data = [go.Bar(
    x = tm.index,
    y = tm/1000 # better to divide the wages(in K) by 1000 to convert them to Millions(M)
    )]
layout = go.Layout(
    autosize=False,
    width=1000,
    height=500,
    title = "Total wages of players of a Nation in the whole game in Million(M) € (Nations having > 100 players)"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

tm = national_team_data[national_team_data['ID']['count']>100]['Wage']['mean'].sort_values(ascending = False)
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=1000,
    height=500,
    title = "Mean wages of players of a Nation in the whole game in Thousands(K) € (Nations having > 100 players)"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

tm = national_team_data[national_team_data['ID']['count']>100]['Age']['mean'].sort_values(ascending = True)
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=1000,
    height=500,
    title = "Mean age of players of a Nation having more than 100 players"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)

nations_more_players = national_team_data[national_team_data['ID']['count']>100]['ID'].sort_values(by = 'count',ascending =  False).index
tm = data.groupby(['Nationality','Overall_position']).agg(['count'])['ID'].unstack()
tm = tm.fillna(0)
tm = tm[tm.index.isin(nations_more_players)]

tm.head()

trace1 = go.Bar(
    x=tm.index,
    y=tm['count']['defender'],
    name='Defenders'
)
trace2 = go.Bar(
    x=tm.index,
    y=tm['count']['midfielder'],
    name='Midfielders'
)
trace3 = go.Bar(
    x=tm.index,
    y=tm['count']['forward'],
    name='Forwards'
)
trace4 = go.Bar(
    x=tm.index,
    y=tm['count']['goalkeeper'],
    name='Goalkeepers'
)
plt_data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    barmode='stack',
    title = "Total number of players position wise playing for a nation",
    width = 1200,
    height = 500
)

fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)
