import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import iqr

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

fig, ax = plt.subplots(1, 2, figsize=(18, 4))

time_val_fraud = dataset_fraud['Time'].values
time_val_not_fraud = dataset_notfraud['Time'].values

sns.distplot(time_val_not_fraud, ax=ax[0], color='r', bins=48)
ax[0].set_title('Districuição de transações normais', fontsize=14)
ax[0].set_xlim([min(time_val_not_fraud), max(time_val_not_fraud)])
ax[0].set_xlabel("Tempo")
ax[0].set_ylabel("Contagem")

sns.distplot(time_val_fraud, ax=ax[1], color='b', bins=48)
ax[1].set_title('Distribuição de transações fraudulentas', fontsize=14)
ax[1].set_xlim([min(time_val_fraud), max(time_val_fraud)])
ax[1].set_xlabel("Tempo")
ax[1].set_ylabel("Contagem")


# remoção de valores discrepantes
upper_limit = dataset.Amount.quantile(0.75) + (1.5 * iqr(dataset.Amount))
print("\n", upper_limit)
print(dataset[dataset.Amount > upper_limit]["Class"].value_counts())

dataset = dataset[dataset.Amount <= 8000]
print("\n", dataset.Class.value_counts())
print("\nPorcentagem de atividade fraudulenta -> {:.2%}".format(
    (dataset[dataset.Class == 1].shape[0] / dataset.shape[0])))

plt.ticklabel_format(style='plain', axis='y')

# ============= Análise de correlação =============
correlacao = dataset.corr()
fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(correlacao, xticklabels=correlacao.columns,
            yticklabels=correlacao.columns, linewidths=.1, cmap="RdBu", ax=ax)
plt.tight_layout()


# Selecionando 3000 mil linhas de usuários nao fraudulentos.
# E criando um conjunto de dados para treino.

not_fraud = dataset[dataset.Class == 0].sample(3000)
fraud = dataset[dataset.Class == 1]

print("\nConjunto de dados para treino ->>")
print(f"\nQuantidade de usuários nao fraudulentos: {len(not_fraud)}"
      f"\nQuantidade de usuários fraudulentos: {len(fraud)}")

new_dataset_Treino = not_fraud.append(
    fraud).sample(frac=1).reset_index(drop=True)
X = new_dataset_Treino.drop(["Class"], axis=1).values
y = new_dataset_Treino.Class.values

p = TSNE(n_components=2, random_state=42).fit_transform(X)
print(p)


color_map = {0: 'red', 1: "blue"}
plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(
        x=p[y == cl, 0],
        y=p[y == cl, 1],
        c=color_map[idx],
        label=cl
    )

plt.xlabel("X no t-SNE")
plt.ylabel("y no t-SNE")
plt.legend(loc="upper left")
plt.title("Visualização dos dados de teste")

# Autoencorder para detecção de fraudes

x_scale = preprocessing.MinMaxScaler().fit_transform(X)
x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]

autoencorder = Sequential()
autoencorder.add(Dense(X.shape[1], activation="tanh"))
autoencorder.add(Dense(100, activation="tanh"))
autoencorder.add(Dense(50, activation="relu"))
autoencorder.add(Dense(50, activation="tanh"))
autoencorder.add(Dense(100, activation="tanh"))
autoencorder.add(Dense(X.shape[1], activation="relu"))

autoencorder.compile(optimizer="adadelta", loss="mse")


# Treinando o autoencoder
autoencorder.fit(x_norm, x_norm,
                 batch_size=256, epochs=10,
                 shuffle=True, validation_split=0.20)

hidden_representation = Sequential()
hidden_representation.add(autoencorder.layers[0])
hidden_representation.add(autoencorder.layers[1])
hidden_representation.add(autoencorder.layers[2])

norm_hid_rep = hidden_representation.predict(x_norm)
fraud_hid_rep = hidden_representation.predict(x_fraud)

rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis=0)
y_n = np.zeros(norm_hid_rep.shape[0])
y_f = np.ones(fraud_hid_rep.shape[0])
rep_y = np.append(y_n, y_f)


p = TSNE(n_components = 2, random_state = 42).fit_transform(rep_x)

color_map = {0: 'red', 1: "blue"}
plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(
        x=p[y == cl, 0],
        y=p[y == cl, 1],
        c=color_map[idx],
        label=cl
    )
    
plt.xlabel("X no t-SNE")
plt.ylabel("y no t-SNE")
plt.legend(loc="upper left")
plt.title("t-SNE -> Visualização do teste")

plt.show()
