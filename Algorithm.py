import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense

import warnings
from warnings import simplefilter

np.random.seed(42)

simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset = pd.read_csv('creditcard.csv', sep=",")

print(dataset.head(10))

# Estatísticas descritivas
print(dataset.Time.describe())

# Verificando os tipos dos dados
print("\nTipos dos dados: \n{}".format(dataset.dtypes))

# ========================== basic EDA ==========================

# gráfico de contagem das classes
fig, ax = plt.subplots(figsize=(6, 4))
ax = sns.countplot(x="Class", data=dataset)
plt.tight_layout()

# Contagem de transações ao longo das horas
dataset_notfraud = dataset[dataset["Class"] == 0]
dataset_fraud = dataset[dataset["Class"] == 1]

fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val_fraud = dataset_fraud['Time'].values
time_val_not_fraud = dataset_notfraud['Time'].values

sns.distplot(time_val_not_fraud, ax=ax[0], color='r', bins = 48)
ax[0].set_title('Districuição de transações normais', fontsize=14)
ax[0].set_xlim([min(time_val_not_fraud), max(time_val_not_fraud)])
ax[0].set_xlabel("Tempo")
ax[0].set_ylabel("Contagem")

sns.distplot(time_val_fraud, ax=ax[1], color='b', bins = 48)
ax[1].set_title('Distribuição de transações fraudulentas', fontsize=14)
ax[1].set_xlim([min(time_val_fraud), max(time_val_fraud)])
ax[1].set_xlabel("Tempo")
ax[1].set_ylabel("Contagem")

plt.ticklabel_format(style='plain', axis='y')
plt.show()
