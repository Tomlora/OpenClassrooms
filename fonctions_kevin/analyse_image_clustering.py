# à faire

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

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import metrics as kmetrics
from keras.layers import *
from keras.models import Model, Sequential
import gensim

import tensorflow_hub as hub

# Bert
import os
import transformers
import time

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.vgg19 import VGG19
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

# fix pour plotly express et Visual Studio Code
import plotly.io as pio
pio.renderers.default = "notebook_connected"

pd.options.mode.chained_assignment = None  # default='warn'

warnings.simplefilter("ignore", DeprecationWarning)

import cv2
import time
from sklearn import cluster, metrics
from os import listdir
from sklearn import manifold, decomposition


# On désactive les logs de TF / Keras

tf.keras.utils.disable_interactive_logging()
    
# --------------------------------------

class SIFT():
    def __init__(self, path_dossier_img, list_photos=None):
        """
        L'algorithme SIFT va détecter des points sur les images, décrits par des descripteurs comportant un certain nombre de données (vecteurs).
        
        Parameters
        ------------
        path_dossier_img = path vers le dossier qui contient les images
        
        list_photos = liste avec le nom des photos + leur format (jpg, png...).
        Si non renseigné, prends les photos de l'ensemble du dossier path
        
        Démarche générale
        ------------------
        1. On crée les descripteurs
        2. On crée les clusters de descripteurs
        3. On crée les histogrammes par image
        4. On réduit les dimensions PCA / T-SNE
        5. Analyse visuelle (T-SNE) selon catégories et clustering
        """
        self.path_dossier_img = path_dossier_img
        if list_photos is None:
            self.liste_photos = [file for file in listdir(self.path_dossier_img)]
        else:
            self.list_photos = list_photos 
        self.path = self.path_dossier_img + self.list_photos
        
    
    def one_descripteur(self, n):
        """Détermination et affichage des descripteurs SIFT d'une image
        
        Parameters
        ------------
        
        n = numero de l'image de la liste
        
        
        Return
        ------------
        kp = keypoints The detected keypoints. A 1-by-N structure array with the following fields:
        - pt coordinates of the keypoint [x,y]
        - size diameter of the meaningful keypoint neighborhood
        - angle computed orientation of the keypoint (-1 if not applicable); it's in [0,360) degrees and measured relative to image coordinate system (y-axis is directed downward), i.e in clockwise.
        - response the response by which the most strong keypoints have been selected. Can be used for further sorting or subsampling.
        - octave octave (pyramid layer) from which the keypoint has been extracted.
        - class_id object class (if the keypoints need to be clustered by an object they belong to).
        
        image = l'image en noir et blanc
        
        des = descripteurs
        """
        self.sift = cv2.SIFT_create()
        plt.subplot(130 + 1 + 1)
        self.image_original = imread(self.path[n])
        self.image = cv2.imread(self.path[n],0) # convert in gray
        self.image = cv2.equalizeHist(self.image)   # equalize image histogram
        self.kp, self.des = self.sift.detectAndCompute(self.image, None)
        self.img=cv2.drawKeypoints(self.image,self.kp,self.image)
        plt.imshow(self.image_original)
        plt.subplot(130 + 1 + 2)
        plt.imshow(self.img)
        plt.show()
        console = Console()
        
        # print("Descripteurs : ", self.des.shape)
        console.print(f"L'image contient {self.des.shape[0]} descripteurs. Chaque descripteur est un vecteur de longueur {self.des.shape[1]}", style="green")
        # print(self.des)
        
    def all_descripteur(self, n_features=500):
        """ Créer les descripteurs de chaque image
        
        - Chaque image est passé en gris et egaliser.
        - Création d'une liste de descripteurs par image (sift_keypoints_by_img) qui sera utilisée
        pour réaliser les histogrammes par image
        - Création d'une liste de descripteurs pour l'ensemble des images (sift_keypoints_all) qui sera utilisée
        pour créer les clusters de descripteurs
        
        Parameters
        -------------
        n_features : Nombre de features
        
        Return
        --------------
        sift_keypoints_by_img : Liste de descripteurs par image
        sift_keypoints_all : Liste de descripteurs pour l'ensemble des images
        """
        self.sift_keypoints = []
        self.temps1=time.time()
        self.sift = cv2.SIFT_create(n_features)

        for image_num in range(len(self.list_photos)) :
            if image_num%100 == 0 : print(f"{image_num} / {len(self.list_photos)}")
            self.image = cv2.imread(self.path[image_num],0) # convert in gray
            # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            self.res = cv2.equalizeHist(self.image)   # equalize image histogram
            self.kp, self.des = self.sift.detectAndCompute(self.res, None)
            self.sift_keypoints.append(self.des)

        self.sift_keypoints_by_img = np.asarray(self.sift_keypoints)
        self.sift_keypoints_all    = np.concatenate(self.sift_keypoints_by_img, axis=0)

        print()
        print("Nombre de descripteurs : ", self.sift_keypoints_all.shape)
        
        console = Console()

        self.duration1=time.time()-self.temps1
        console.print("temps de traitement SIFT descriptor : ", "%15.2f" % self.duration1, "secondes", style="yellow")
        
    def clustering_kmeans(self, k:int=0):
        """ Création des clusters de descripteurs.
        
        On va regrouper les descripteurs par clusters (Kmeans).
        Ce regroupement de descripteur peut être comparé au bag of words pour des phrases.
        
        Utilisation de MiniBatchKmeans pour obtenir des temps de traitement raisonnables
        Entrainement sur la Liste de descripteurs pour l'ensemble des images
        
        Parameters
        ------------
        k = Nombre de clusters (Si non précisé : racine carré du nombre total de descripteurs)
        
        ### Choix du Nombre de clusters
        
        nb_clusters (si non précisé) = racine carré du nombre total de descripteurs (Nombre maximal)
        
        Autre possibilité : prendre le nombre de catégories à traiter et multiplier par 10 (Nombre minimal)
        
        On peut aussi prendre une valeur intermédiaire entre les deux
        
        Return
        -----------
        
        Kmeans fit
        """
        # Determination number of clusters
        self.temps1=time.time()

        if k == 0:
            self.k = int(round(np.sqrt(len(self.sift_keypoints_all)),0))
        else:
            self.k = k*10
        
        console = Console()
        
        console.print(f"Nombre de clusters estimés : {self.k}", style="purple")
        console.print(f"Création de {self.k} clusters de descripteurs ...", style="green")

        # Clustering
        self.kmeans = cluster.MiniBatchKMeans(n_clusters=self.k, init_size=3*self.k, random_state=0)
        self.kmeans.fit(self.sift_keypoints_all)

        self.duration1=time.time()-self.temps1
        print("temps d'entrainement kmeans : ", "%15.2f" % self.duration1, "secondes")
        
        
    def features_img(self):
        """ Création des features des images.
        
        Pour chaque image :
        - Prédiction des numéros de cluster de chaque image avec ses descripteurs
        - Création d'un histogramme = comptage pour chaque numéro de cluster du nombre de descripteurs de l'image
        
        Features d'une image = Histogramme d'une image = Comptage pour une image du nombre de descripteurs par cluster
        
        
        Return
        ---------------
        
        im_features = features de la matrice des img
        """
        # Creation of histograms (features)
        self.temps1=time.time()

        def build_histogram(kmeans, des, image_num):
            res = kmeans.predict(des)
            hist = np.zeros(len(kmeans.cluster_centers_))
            nb_des=len(des)
            if nb_des==0 : print("problème histogramme image  : ", image_num)
            for i in res:
                hist[i] += 1.0/nb_des #comptage. Sorte de bag of word manuel pondéré par le nombre de descripteurs car les img ont un nombre de descripteurs très différent
            return hist


        # Creation of a matrix of histograms
        self.hist_vectors=[]

        for i, image_desc in enumerate(self.sift_keypoints_by_img) :
            if i%100 == 0 : print(f"{i} / {len(self.sift_keypoints_by_img)}")  
            self.hist = build_histogram(self.kmeans, image_desc, i) #calculates the histogram
            self.hist_vectors.append(self.hist) #histogram is the feature vector

        self.im_features = np.asarray(self.hist_vectors)

        self.duration1=time.time()-self.temps1
        print("temps de création features de la matrice image : ", "%15.2f" % self.duration1, "secondes")
        
        
    def features_pca(self, n_components=0.99):
        """* La réduction PCA permet de créer des features décorrélées entre elles, et de diminuer leur dimension, tout en gardant un niveau de variance expliquée élevé (99%)
        * L'impact est une meilleure séparation des données via le T-SNE et une réduction du temps de traitement du T-SNE
        
        Parameters
        ------------
        n_components = nombre de composants du PCA (0.99 par defaut)
        
        Return
        ------------
        Features traités via PCA"""
        
        console = Console()
        console.print(f"Dimensions dataset avant réduction PCA : {self.im_features.shape}", style="purple")
        self.pca = decomposition.PCA(n_components=n_components)
        self.feat_pca= self.pca.fit_transform(self.im_features)
        console.print(f"Dimensions dataset après réduction PCA : {self.feat_pca.shape}", style="green")
        
        
# ------------------------

class TransfertLearning():    
    def __init__(self, model, path_dossier_img, list_photos, target_size=(224,224)):
        """        
        L'entraînement des modèles de réseaux de neurones convolutifs profonds sur de très grands ensembles de données peut prendre des jours, voire des semaines.

        Une façon de raccourcir ce processus est de réutiliser des modèles pré-entraînés qui ont été développés pour des ensembles de données de référence standard, tels que les tâches de reconnaissance d'images. 
        
        Les modèles les plus performants peuvent être téléchargés et utilisés directement, ou intégrés dans un nouveau modèle pour nos propres problèmes.
        
        L'apprentissage par transfert fait généralement référence à un processus dans lequel un modèle formé sur un problème est utilisé d'une certaine manière sur un second problème connexe.

        Dans l'apprentissage profond, l'apprentissage par transfert est une technique par laquelle un modèle de réseau neuronal est d'abord formé sur un problème similaire au problème à résoudre. 

        Une ou plusieurs couches du modèle formé sont ensuite utilisées dans un nouveau modèle formé sur le problème en question.
        
        L'apprentissage par transfert présente l'avantage de réduire le temps de formation d'un modèle de réseau neuronal et peut entraîner une erreur de généralisation plus faible.

        Les couches réutilisées peuvent être utilisés comme point de départ du processus de formation et adaptés en réponse au nouveau problème. 
        
        Cela peut être utile lorsque le premier problème connexe comporte beaucoup plus de données étiquetées que le problème d'intérêt et que la similitude de la structure du problème peut être utile dans les deux contextes.
        
        Ces modèles peuvent être utilisés comme base pour l'apprentissage par transfert.

        Avantages :
        ---------------------------------------
        Fonctions utiles apprises : Les modèles ont appris à détecter les caractéristiques génériques des photographies, étant donné qu'ils ont été entraînés sur plus de 1 000 000 d'images pour 1 000 catégories.
        
        Performances de pointe : Les modèles ont atteint une performance de pointe et restent efficaces dans la tâche spécifique de reconnaissance d'images pour laquelle ils ont été développés.
        
        Facilité d'accès : Les modèles sont fournis sous forme de fichiers téléchargeables gratuitement et de nombreuses bibliothèques fournissent des API pratiques pour télécharger et utiliser directement les modèles.
        
        Notes
        ----------
        Lors du chargement d'un modèle donné, l'argument "include_top" peut être défini à False, auquel cas les couches de sortie entièrement connectées du modèle utilisé pour faire des prédictions ne sont pas chargées, ce qui permet d'ajouter une nouvelle couche de sortie et de l'entraîner.
        
        En revanche, input_tensor doit être renseigné avec par exemple :
        >>> input_tensor = Input(shape=(240, 240, 3))
        
        Src : https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
        
        Parameters
        ----------
        
        model :
        - VGG16 : par Visual Graphic Group à Oxford : https://arxiv.org/abs/1409.1556 : Format des img 224x224
        - VGG19
        - RESNET50 : par des chercheurs de Microsoft : https://arxiv.org/abs/1512.03385 Format des images 224x244
        - InceptionV3 : Par des chercheurs de Google : https://arxiv.org/abs/1512.00567  Format des images 299x299
        
        path_dossier : url vers le dossier des img
        
        list_photos : liste avec les noms des photos + format (.jpg, .png ..)
        
        target_size :class:`tuple` par défaut > (224,224)
        
        """
        if model == "VGG16":
            self.model = VGG16()
        elif model == "VGG19":
            self.model = VGG19()
        elif model == "RESNET50":
            self.model = ResNet50V2()
        elif model == "InceptionV3":
            self.model = InceptionV3()
            
        self.target_size=target_size
        self.path_dossier_img = path_dossier_img
        self.list_photos = list_photos
        self.path = self.path_dossier_img + self.list_photos

        
    def clustering_modele_entraine(self, df, colonne_clustering, n=0):
        
        """ Un algorithme pré-entrainé peut être utilisé directement pour classifier des images.
        
        Tout d'abord, la photographie doit être chargée et transformée en un carré de 224×224 ou 299x299 (principalement) pixels, comme prévu par le modèle, et les valeurs des pixels doivent être mises à l'échelle comme prévu par le modèle. 
        
        Le modèle fonctionne sur un tableau d'échantillons, donc les dimensions d'une image chargée doivent être étendues de 1, pour une image de 224×224 pixels et trois canaux :
        >>> shape(1, 224, 224, 3)
        
        
        
        Parameters
        ------------------
        
        df = DataFrame
        
        colonne_clustering :class:`Series` = Colonne avec les vrais labels
        
        n = Numero de l'image pour un exemple illustré
        
        Return
        ------------------
        
        Exemple illustré pour l'image n
        
        df_model = DataFrame avec 4 colonnes contenant toutes les images :
        - Nom des images
        - Catégorie devinée par l'algorithme
        - Vraie catégorie
        - Score de l'algorithme sur cette catégorie 
        """
        self.n = n
        
        self.liste_name = []
        self.liste_categorie = []
        self.liste_score = []
        self.df_clustering = df[colonne_clustering] #data_img['categorie']
         
        # Exemple avec une image n :
        self.img = load_img(self.path[n], target_size=self.target_size)  # Charger l'image
        self.img = img_to_array(self.img)  # Convertir en tableau numpy
        self.img = self.img.reshape((1, self.img.shape[0], self.img.shape[1], self.img.shape[2]))  # Créer la collection d'images (un seul échantillon). Les modeles attendent une taille Numpy array précise (comme 224x224)
        # Ici nous avons 4 dimensions (echantillon, ligne, colonne, channel)
        self.img = preprocess_input(self.img)  # Prétraiter l'image comme le veut VGG-16
        
        # On fait la prédiction pour l'image n :
        self.y = self.model.predict(self.img)
        self.df_keras = pd.DataFrame(decode_predictions(self.y, top=3)[0]) # top 3 de la prédiction
        
        # On montre l'image n
        self.img_n = cv2.imread(self.path[n])
        plt.imshow(self.img_n)
        plt.show()
        
        # Graphique avec les résultats pour la prédiction de l'image n
        self.fig = px.histogram(self.df_keras, x=2, y=1, color=2).update_yaxes(categoryorder="total ascending")
        self.fig.update_layout(showlegend=False)
        self.fig.show()
        
        # On fait désormais le traitement pour toutes les images...
        
        for i in range(len(self.list_photos)):
    
            self.img = load_img(self.path[i], target_size=self.target_size)  # Charger l'image
            self.img = img_to_array(self.img)  # Convertir en tableau numpy
            self.img = self.img.reshape((1, self.img.shape[0], self.img.shape[1], self.img.shape[2]))  # Créer la collection d'images (un seul échantillon). Les modeles attendent une taille Numpy array précise (comme 224x224).
            # Ici nous avons 4 dimensions (echantillon, ligne, colonne, channel)
            self.img = preprocess_input(self.img)  # Prétraiter l'image comme le veut VGG-16

            # La prédiction
            self.y = self.model.predict(self.img)
            
            # On met les résultats dans une liste ...
            
            self.liste_name.append(self.list_photos[i]) # nom de l'image
            self.liste_categorie.append(decode_predictions(self.y, top=1)[0][0][1]) # catégorie devinée
            self.liste_score.append(decode_predictions(self.y, top=1)[0][0][2]) # score
            
        # On crée le dataframe avec l'ensemble
        self.df_model = pd.DataFrame([self.liste_name, self.liste_categorie, self.liste_score])
        self.df_model = self.df_model.transpose()
        self.df_model.insert(2, "Truth Categorie", self.df_clustering, True)
        self.df_model = self.df_model.rename(columns={0 : "Image", 1 : "Categorie Algo", 2 : "Indice de confiance"})
        
        
    def featuring_extract(self, couche=-2):
        # Src : https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
        """Les modèles peuvent être téléchargés et utilisés comme modèles d'extraction de caractéristiques (features). 
        
        Ici, la sortie du modèle d'une couche antérieure à la couche de sortie du modèle est utilisée comme entrée d'un nouveau modèle de classificateur (kmeans par exemple).

        Rappelons que : 
        - Les couches convolutionnelles les plus proches de la couche d'entrée du modèle apprennent des caractéristiques de bas niveau telles que des lignes
        - Les couches du milieu de la couche apprennent des caractéristiques abstraites complexes qui combinent les caractéristiques de bas niveau extraites de l'entrée
        - Les couches plus proches de la sortie interprètent les caractéristiques extraites dans le contexte d'une tâche de classification.

        Fort de cette compréhension, il est possible de choisir un niveau de détail pour l'extraction des caractéristiques d'un modèle pré-entraîné existant. 
        
        C'est pourquoi on ne va pas supprimer que les deux dernières couches.
        
        Par exemple, si une nouvelle tâche est très différente de la classification d'objets dans des photographies (par exemple, différente d'ImageNet), alors peut-être que la sortie du modèle pré-entraîné après quelques couches serait appropriée. 
        
        Si une nouvelle tâche est très similaire à la classification d'objets dans des photographies, on peut utiliser la sortie de couches beaucoup plus profondes dans le modèle, ou même la sortie de la couche entièrement connectée avant la couche de sortie."""
        self.data_algo = {}
        # Now we can load the VGG model and remove the output layer manually. 
        # This means that the new final layer is a fully-connected layer  with 4,096 output nodes. 
        # This vector of 4,096 numbers is the feature vector that we will use to cluster the images.
        self.layers = self.model.layers
        self.model_perso = Model(inputs=self.model.inputs, outputs=self.layers[couche].output) # on prend le modèle d'input du modèle sélectionné, et on prend l'output de la deuxième dernière couche des outputs (pour éviter la classification qui est faite par les dernières couches)
        # On fait le traitement de toutes les images
        for i in range(len(self.list_photos)):
    
            self.img = load_img(self.path[i], target_size=self.target_size)  # Charger l'image
            self.img = img_to_array(self.img)  # Convertir en tableau numpy
            self.img = self.img.reshape((1, self.target_size[0], self.target_size[1], 3))  # Créer la collection d'images (un seul échantillon) Les modeles attendent une taille Numpy array précise (comme 224x224).
            # Ici nous avons 4 dimensions (echantillon, ligne, colonne, channel)
            self.img = preprocess_input(self.img)  # Prétraiter l'image comme le veut VGG-16

            # La prédiction
            self.y = self.model_perso.predict(self.img)
            
            # on met dans un dictionnaire l'ensemble des informations (nom de l'img / features)
            
            self.data_algo[self.list_photos[i]] = self.y
            
        self.filenames = np.array(list(self.data_algo.keys())) # Nom des images
        self.features = np.array(list(self.data_algo.values())) # Valeurs de l'array.
        
        console = Console()
        console.print(f' Dimensions des features : {self.features.shape}')
        
    
    def featuring_pca(self, redimension:tuple, n_components=100):
        """
        Comme notre vecteur de caractéristiques a énormément de dimension, il vaut mieux le réduire à un nombre beaucoup plus petit. 
        
        Nous ne pouvons pas simplement raccourcir la liste en la découpant ou en utilisant un sous-ensemble de celle-ci, car nous perdrions des informations.
        
        On fait donc un PCA pour garder autant de dimensions que possibles.

        Pour cela, il faut une array en 2D, donc faire également une redimension pour remplir cette condition
        
        Parameters
        ------------
        
        Redimension :class:`tuple`: Tuple de redimension.
        - Si dimension (sample, 1, 2048) -> (-1,2048)
        - Si dimension (sample, 1, 4096) -> (-1, 4096)
        
        n_components = composants pour PCA (Par défaut : 100)
        
        Return
        -----------
        
        features = Features avec la nouvelle shape
        
        features_pca = features avec la nouvelle shape + pca
        
        """
        self.redimension = redimension

        self.features = self.features.reshape(self.redimension)
        console = Console()
        console.print(f'Nouvelles dimensions : {self.features.shape}') 
        
        self.pca = decomposition.PCA(n_components=n_components)
        self.features_pca = self.pca.fit_transform(self.features)