import time

from linear_assignement import linear_assignment

import warnings
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from matplotlib.image import imread

import numpy as np



from sklearn.metrics import confusion_matrix


import nltk

from sklearn import cluster, metrics
from sklearn import manifold, decomposition

import tensorflow as tf
import keras
from keras import backend as K

import torch
import seaborn as sns



import tensorflow_hub as hub

# Bert


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')



# fix pour plotly express et Visual Studio Code
import plotly.io as pio
pio.renderers.default = "notebook_connected"

pd.options.mode.chained_assignment = None  # default='warn'

warnings.simplefilter("ignore", DeprecationWarning)


import time
from sklearn import cluster, metrics

from sklearn import manifold, decomposition


def preparation_variable_clustering(df, colonne):
    """Créer deux listes avec la variable discriminante qui servira à l'ARI.
    - La première contient les valeurs uniques pour le clustering
    - La seconde est une liste contenant les valeurs de la variable en format numérique.

    Parameters
    ----------

    df : :class:`DataFrame`

    colonne : :class:`Series` : Colonne du DataFrame qui est la variable discriminante

    Return
    ----------
    l_cat : Liste avec les valeurs uniques
    y_cat_num : Liste avec les valeurs str en int
    """
    l_cat = list(set(df[colonne])
                 )  # variable discriminante pour les graphiques
    # print("catégories : ", l_cat)

    # on transforme les catégories en valeurs numériques
    y_cat_num = [(l_cat.index(df.iloc[i][colonne])) for i in range(len(df))]

    l_num_cat = "categories : " + \
        str([f" {categorie} : {l_cat.index(categorie)}" for categorie in l_cat])

    print(l_num_cat)
    return l_cat, y_cat_num

# Calcul Tsne, détermination des clusters et calcul ARI entre vrais catégorie et n° de clusters


def ARI_fct(features, l_cat, y_cat_num, perplexity=30, learning_rate=200, tsne:bool=True):
    """
    Calcul Tsne, détermination des clusters et calcul ARI entre vrais catégorie et n° de clusters

    Parameters
    ----------

    Features : :class:`transformé par un algorithme` : Features transformé par id-tf, cvc, bert, USE, Word2Vec.predict() ....

    l_cat : :class:`List` : Liste avec les valeurs uniques qui serviront de variables discriminantes

    y_cat_num : :class:`List` : Liste avec les valeurs transformées en nombre. Nécessaire pour comparaison Kmeans et réelles (ARI)
    
    tsne : :class:`bool` : (True par defaut) Applique un TSNE ou non 


    """

    time1 = time.time()
    num_labels = len(l_cat)
    if tsne is True:
        tsne = manifold.TSNE(n_components=2, perplexity=perplexity, n_iter=2000,
                            init='random', learning_rate=learning_rate, random_state=42)
        X_tsne = tsne.fit_transform(features)
    else:
        X_tsne = features

    # Détermination des clusters à partir des données après Tsne
    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)
    cls.fit(X_tsne)
    ARI = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_), 4)
    time2 = np.round(time.time() - time1, 0)
    print("ARI : ", ARI, "time : ", time2)

    return ARI, X_tsne, cls.labels_


# visualisation du Tsne selon les vraies catégories et selon les clusters
def TSNE_visu_fct(X_tsne, l_cat, y_cat_num, labels, ARI):
    """ 
    Visualisation du Tsne selon les vraies catégories et selon les clusters

    Parameters
    -----------

    X_tsne : :class:`Données T-SNE`

    l_cat : :class:`List` : Liste avec les valeurs uniques qui serviront de variables discriminantes

    y_cat_num : :class:`List`: Liste avec les valeurs transformées en numéro

    labels : :class:`List` : Liste avec les labels générés par Kmeans

    ARI : >>> ARI = np.round(metrics.adjusted_rand_score(y_cat_num, labels),4)

    """
    fig = plt.figure(figsize=(15, 6))
    
    # --------- Scatterplot : catégories réelles

    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[
              0], labels=l_cat, loc="best", title="Categorie")
    plt.title('Catégories réelles')
    
    # --------- Scatterplot : catégories Kmeans

    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(
        labels), loc="best", title="Clusters")
    plt.title('K-Means Clustering')

    plt.show()
    
    # --------------- Matrice de confusion
    # source : https://smorbieu.gitlab.io/fr/accuracy-clustering-classification-non-supervis%C3%A9e.html

    cm = confusion_matrix(y_cat_num, labels)

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    
    s = sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
    s.set(xlabel="Kmeans label", ylabel="Truth label")
    
    plt.show()
    
    # -------------- Répartition 

    fig = px.pie(names=labels)
    fig.show()