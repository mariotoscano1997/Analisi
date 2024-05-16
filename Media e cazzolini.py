# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:47:39 2024

@author: tosca
"""
from matplotlib import pyplot as plt

#importare database
import pandas as pd
data = pd.read_csv("http://antoninofurnari.it/downloads/height_weight.csv")
#mostra le info
data.info()

#visualizzo testa
print(data.head())

sex_counts = data['sex'].value_counts()
print("------SESSO----")
print(sex_counts)

#vedere quanti tipi di variabili ci sono
print("------VALORI UNICI----")
print(data['sex'].nunique())
print(data['height'].nunique())
print(data['weight'].nunique())

#vediamo le frequenze ass di weight perchè ci sono pochi valori 
print("------ALTEZZA----")
height_counts = data['height'].value_counts()
print(height_counts)

print("Altezza più frequente:",height_counts.iloc[0])
print("Altezza meno frequente:",height_counts.iloc[-1])

#Ordina per indice
print(data['height'].value_counts().sort_index())

#per il grafico 
data['height'].value_counts().sort_index().plot.bar(figsize=(18,6))
plt.grid()
plt.show()

#frequenze relative 
print(data['height'].value_counts().sort_index()/len(data['height'].dropna()))
#alternativamente si può usare stessa cosa
print(data['height'].value_counts(normalize=True).sort_index())

#per graficare 
data['height'].value_counts(normalize=True).sort_index().plot.bar(figsize=(18,6))
plt.grid()
plt.show()

#mettiamo gli istogrammi di uomini e donne a confronto per visualizzare le info
plt.figure(figsize=(18,6))
pmf_height_m = data[data['sex']=='M']['height'].value_counts(normalize=True).sort_index()
pmf_height_f = data[data['sex']=='F']['height'].value_counts(normalize=True).sort_index()
#sommiamo e sottraiamo 0.2 agli indici per "spostare" le barre e renderle
#visibili quando sovrappose. Inoltre impostiamo alpha=0.9 per rendere le barre
#parzialmente trasparenti
plt.bar(pmf_height_m.index+0.2, pmf_height_m.values, width=1, alpha=0.9)
plt.bar(pmf_height_f.index-0.2, pmf_height_f.values, width=1, alpha=0.9)
plt.xticks(data['height'].unique(), rotation='vertical')
plt.legend(['M','F']) #mostriamo una legenda
plt.grid()
plt.show()

#calcoliamo la ecdf
ecdf_weight_m = data[data['sex']=='M']['weight'].value_counts(normalize=True).sort_index().cumsum()
ecdf_weight_f = data[data['sex']=='F']['weight'].value_counts(normalize=True).sort_index().cumsum()

#ed adesso la grafichiamo
plt.figure(figsize=(12,8))
plt.plot(ecdf_weight_m.index, ecdf_weight_m.values)
plt.plot(ecdf_weight_f.index, ecdf_weight_f.values)
plt.legend(['M','F'])
plt.grid()
plt.show()

'''
TITANIC
'''

#Analizziamo titanic
titanic = \
pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv',
                     index_col='PassengerId')
titanic.info()

a=titanic[titanic['Sex']=='male']['Age'].value_counts(normalize=True).sort_index().cumsum()
b=titanic[titanic['Sex']=='female']['Age'].value_counts(normalize=True).sort_index().cumsum()

plt.figure(figsize=(12,8))
plt.plot(a.index,a.values)
plt.plot(b.index,b.values)
plt.grid()
plt.legend(['M','F'])
plt.show()

#istogramma 
plt.figure(figsize=(12,6))
_,edges,_=plt.hist(data['weight'], width=10.8)
plt.xticks(edges)
plt.grid()
plt.show()

#strugles and rice 
import numpy as np
bins_struges=int(3.3*np.log(len(data['weight'])))
bins_rice=int(2*len(data['weight'])**(1/3))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title('Struges ({} bins)'.format(bins_struges))
plt.hist(data['weight'], bins=bins_struges)
plt.grid()
plt.subplot(1,2,2)
plt.title('Rice ({} bins)'.format(bins_rice))
plt.grid()
plt.hist(data['weight'], bins=bins_rice)
plt.show()


#istogramma di densità parametro density = true
plt.figure(figsize=(12,6))
valori,bordi,_=plt.hist(data['weight'], bins=[40,50,60,70,80,90,100,110,120,130], 
                   width=9.5, density=True)
plt.xticks(bordi)
plt.grid()
plt.show()

#plottare la stima di densita come segue

data['weight'].plot.density(figsize=(12,6))
plt.grid()
plt.show()

#confrontare gli istogrammi 
plt.figure(figsize=(12,6))
plt.hist(data[data['sex']=='F']['weight'], width=7, alpha=0.9)
plt.hist(data[data['sex']=='M']['weight'], width=7, alpha=0.9)
plt.legend(['F','M'])
plt.grid()
plt.show()
#utilizzando groupby è possibile aggregare i grafici
data.groupby('sex')['weight'].plot.hist(width=7, alpha=0.9, density=True, figsize=(12,6))
data.groupby('sex')['weight'].plot.density()
plt.legend()
plt.grid()
plt.show()

#barPlot 
titanic.groupby('Sex')['Sex'].count().plot.bar()
plt.ylabel('Number of Passengers')
plt.show()

#barplot per classi 
titanic.groupby('Pclass')['Age'].mean().plot.bar()
plt.ylabel('Age')
plt.show()


#calcoliamo i sopravvisuti 
print(pd.crosstab(titanic['Pclass'], titanic['Survived']))

#ed adesso vediamo le frequenze relative
print(pd.crosstab(titanic['Pclass'], titanic['Survived'], normalize='index'))

#adesso visualizziamo i plot relativi alle freq. relative
pd.crosstab(titanic['Pclass'], titanic['Survived'], normalize='index').plot.bar()
plt.ylabel('Survived %')
plt.show()

#visione tramite stack bar 
pd.crosstab(titanic['Pclass'], titanic['Survived'], normalize='index').plot.bar(stacked=True)
plt.ylabel('%')
plt.show()

#adesso identifichiamo la suddivisione tramite maschi e femmine per classi 
pd.crosstab([titanic['Pclass'], titanic['Sex']], titanic['Survived'], normalize='index').plot.bar(stacked=True)
plt.ylabel('%')
plt.show()

#grafico a torta
titanic.groupby('Sex')['Survived'].sum().plot.pie()
plt.xlabel('Sex')
plt.axis('equal')
plt.show()















































