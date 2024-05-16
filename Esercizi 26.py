# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:56:24 2024

@author: tosca
"""

'''
ESERCIZI CAPITOLO 26
'''
import pandas as pd
from matplotlib import pyplot as plt
titanic = \
pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv',
                     index_col='PassengerId')

'''
Considerando il dataset Titanic, si mostrino i diagrammi a barre delle frequenze assolute
 e relative dei valori della colonna Pclass
'''
titanic['Pclass'].value_counts().sort_index().plot.bar(figsize=(18,6))
plt.grid()
plt.show()

titanic['Pclass'].value_counts(normalize=True).sort_index().plot.bar(figsize=(18,6))
plt.grid()
plt.show()

'''
Scegliendo a caso tra i passeggeri, qual è la probabilità che egli si sia imbarcato in seconda classe
'''
print("Scegliendo a caso tra i passeggeri, qual è la probabilità che egli si sia imbarcato in seconda classe?")
valore = titanic['Pclass'].value_counts(normalize=True).sort_index()
print("{val}%".format(val=valore[2]))

'''
Considerando il dataset Titanic, si mostri l’istogramma delle età dei passeggeri imbarcati in prima classe.
Si utilizzi un criterio opportuno per scegliere il numero di bin.
'''

first=titanic[titanic['Pclass']==1]
bins_rice=int(2*len(first['Age'])**(1/3))
plt.figure(figsize=(12,4))
plt.title('Rice ({} bins) Eta passegeri prima classe'.format(bins_rice))
plt.grid()
plt.hist(first['Age'], bins=bins_rice)
plt.show()

'''
Si modifichi il plot dell’esercizio precedente per costruire un istogramma che permetta di rispondere
alla domanda “se scelgo casualmente un passeggero imbarcato in prima classe, qual è la probabilità che 
la sua età sia compresa tra 20 e 30 anni?
'''

plt.figure(figsize=(12,6))
valori,bordi,_=plt.hist(first['Age'], bins=[0,10,20,30,40,50,60,70,80,90,100], 
                   width=10, density=True)
plt.title('eta raggruppata per 10 anni ')
plt.xticks(bordi)
plt.grid()
plt.show()
print("se scelgo casualmente un passeggero imbarcato in prima classe, qual è la probabilità che la sua età sia compresa tra 20 e 30 anni?")
print(valori[2])

'''
Considerando il dataset Titanic, si confrontino le distribuzioni cumulative delle età 
dei passeggeri imbarcati nelle varie classi. Le distribuzioni sono simili?
In quale classe sono imbarcati i soggetti più giovani?
'''

ecdf_First_C = titanic[titanic['Pclass']==1]['Age'].value_counts(normalize=True).sort_index().cumsum()
ecdf_Second_C = titanic[titanic['Pclass']==2]['Age'].value_counts(normalize=True).sort_index().cumsum()
ecdf_Third_c = titanic[titanic['Pclass']==3]['Age'].value_counts(normalize=True).sort_index().cumsum()
plt.figure(figsize=(12,8))
plt.plot(ecdf_First_C.index, ecdf_First_C.values)
plt.plot(ecdf_Second_C.index, ecdf_Second_C.values)
plt.plot(ecdf_Third_c.index, ecdf_Third_c.values)
plt.legend(['1 classe','2 classe','3 classe'])
plt.grid()
plt.show()

'''
Considerando il dataset Titanic, 
si mostrino con uno stacked bar plot le percentuali di passeggeri appartenenti alle tre classi
separatamente per i due sessi. Si notano differenze nella ripartizione?'''

pd.crosstab(titanic['Sex'], titanic['Pclass'], normalize='index').plot.bar(stacked=True)
plt.ylabel('%')
plt.show()




































