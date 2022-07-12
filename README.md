# Description des projets OpenClassrooms


## Projet 2 : Analyse de données de système éducatif

Nous disposions d'un dataset regroupant des variables très diversifiées sur les pays du monde entier, concernant par exemple l'éducation ou le niveau économique de chaque pays.
Les données sont [disponibles ici](https://datacatalog.worldbank.org/search/dataset/0038480)

Le but du projet était de déterminer, pour une entreprise souhaitant mettre en place des cours en ligne, les pays avec le plus fort potentiel et ceux auquel l'entreprise devrait opérer en priorité.

### Compétences évaluées :
- Utiliser un notebook Jupyter
- Effectuer une représentation graphique
- Manipuler des données avec des librairies spécialisées
- Mettre en place un environnement Python
- Maitriser les opérations fondamentales du langage Python pour la Data Science.


### Ressources (non-exhaustif) :
- Pandas
- Numpy
- Missingno
- Plotly.express
- Pygal
- Seaborn


## Projet 3 : Concevez une application au service de la santé publique

Nous disposions d'un jeu de données regroupant des produits alimentaires du monde entier, et divers variables [disponible ici](https://world.openfoodfacts.org/data/data-fields.txt)
Ces données sont répartis en 4 thèmes :
- Les informations générales du produit (nom, date de création...)
- Des tags (Catégorie, localisation, origine...)
- Les ingrédients
- Les informations nutritionnelles (100g pour 100g de produit)

Le but du projet était de proposer une application innovante pour répondre à un appel à projet lancé par Santé publique France.

### Compétences évaluées :
- Effectuer une analyse statistique multivariée
- Communiquer ses résultats à l'aide de représentations graphiques lisibles et pertinentes
- Effectuer une analyse statistique univariée
- Effectuer des opératirons de nettoyage sur des données structurées.

### Ressources (non-exhaustif) :
- Pandas
- __Matplotlib__ (LineCollection)
- Missingno
- __Sklearn__ (PCA / KNNImputer / StandardScaler)
- __Pingouin__ pour le calcul de l'Anova
- __Ipywidgets__ pour la présentation du produit final

## Projet 4 : Anticipez les besoins en consommation électrique de batiments

Le but du projet est de déterminer la consommation électrique et les émissions de C02 pour les batiments non destinés à l'habitation dans la ville de Seattle.
Les données sont [disponibles ici](https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking#2015-building-energy-benchmarking.csv)

### Compétences évaluées :
- Mettre en place le modèle d'apprentissage supervisée adapté au problème metier
- Adapter les hyperparamètres d'un algorithme d'apprentissage supervisé afin de l'améliorer
- Transformer les variables pertinentes d'un modèle d'apprentissage supervisé
- Evaluer les performances d'un modèle d'apprentissage supervisé

### Ressources (non-exhaustif) :
- Pandas
- Numpy
- __Sklearn__ (FunctionTransformer, validation_curve, cross_validation, OneHotEncoder, StandardScaler, svm)
- __Sklearn.linear_model__ (LinearRegression, Lasso, Ridge, ElasticNet)
- __Sklearn.metrics__ (mean_absolute_error, mean_squared_error)
- Ast
- __Folium__ (map)
- __Optuna__ pour l'optimisation des paramètres


## Projet 5 : Segmentez des clients d'un site e-commerce

Le but du projet est d'aider une entreprise brésilienne qui propose une solution de vente sur les marketplaces en ligne.
En effet, nous devons leur fournir une segmentation des clients qui pourront servir à des campagnes de publicités.

Les données sont [disponibles ici](https://www.kaggle.com/olistbr/brazilian-ecommerce)

### Compétences évaluées :
- Mettre en place le modèle d'apprentissage non supervisé adapté au problème métier
- Transformer les variables pertinentes d'un modèle d'apprentissage non supervisé
- Adapter les hyperparamètres d'un algorithme non supervisé afin de l'améliorer
- Evaluer les performances d'un modèle d'apprentissage non-supervisé

### Ressources (non-exhaustif):
- Pandas
- Numpy
- __Plotly.express__ & __plotly.graph_objects__
- __Sklearn.cluster__ (KMeans)
- __Sklearn.metrics__ (silhouette_samples, silhouette_score, adjusted_rand_score)
- __PCA__ / __TSNE__


## Projet 6 : Classifiez automatiquement des biens de consommation

Le but du projet est d'aider une entreprise qui souhaite lancer une marketplace e-commerce.

Afin d'y parvenir, elle souhaite automatiser l'attribution de la catégorie de chaque article.

Notre rôle est de réaliser une étude de faisabilité d'un moteur de classification d'articles, en se basant sur une image et une description.

### Contraintes du projet :

#### Classification texte :
- Deux approches de type bag-of-words (__comptage simple de mots__ et __Tf-idf__)
- Trois approches de type sentence embedding (__Word2Vec__ / __BERT__ / __USE__)

### Classification image
- Un algorithme de type __SIFT__ / __ORB__ / __SURF__
- Un algorithme de type __CNN Transfert Learning__

### Compétences évaluées :

- Prétraiter des données textes pour obtenir un jeu de données exploitable
- Prétraiter des données image pour obtenir un jeu de données exploitable
- Représenter graphiquement des données à grandes dimensions
- Mettre en oeuvre des techniques de réduction de dimension

### Ressources (non-exhaustif) :
- Pandas
- Numpy
- __Sklearn.cluster__ (KMeans)
- __Sklearn.feature_extraction__ (CountVectorizer, TfidfVectorizer)
- __NTLK__ (tokenize, stem, lemmatizer, stopwords)
- __Tensorflow__ (BERT, USE)
- __Keras__ (Preprocessing / Layers / Metrics / Models / Applications (VGG16, VGG19, Resnetv50, InceptionV3 pour du transfert learning))
- pyLDAvis (algorithme pour identifier des thèmes et les mots clés de chaque thème dans un corpus de texte)
- PCA / TSNE
- __OpenCV__ (pour SIFT)
- Torch

