import tensorflow as tf
import keras
import torch

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
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

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

from linear_assignement import linear_assignment

import tensorflow_hub as hub

# Bert
import os
import transformers
import time

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')



# fix pour plotly express et Visual Studio Code
import plotly.io as pio
pio.renderers.default = "notebook_connected"

pd.options.mode.chained_assignment = None  # default='warn'

warnings.simplefilter("ignore", DeprecationWarning)


# On désactive les logs de TF / Keras

tf.keras.utils.disable_interactive_logging()

def verification_gpu():
    """ Verification que le gpu est disponible"""
    print('Tensorflow : ')
    print(tf.__version__)
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    print(tf.test.is_built_with_cuda())
    print(tf.test.gpu_device_name())
    print(tf.config.list_physical_devices('GPU'))

    print('Torch : ')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)



# -------------------------------

from nltk.corpus import stopwords

def compte_nombre_mots_colonne(df, colonne):
    """
    Compte le nombre de mots par ligne dans une colonne

    Parameters
    ----------

    df : :class:`dataframe` : Le dataframe

    colonne : :class:`colonne dataframe` : Colonne du dataframe à compter

    Return :
    ----------
    Le nombre de mots pour chaque ligne

    Exemples
    ----------
    data['length_bow'] = compte_nombre_mots_colonne(data, 'sentence_bow')
    """

    return df[colonne].apply(lambda x: len(word_tokenize(x)))



# cours associé : https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4855001-representez-votre-corpus-en-bag-of-words
# https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4470548-recuperez-et-explorez-le-corpus-de-textes

def transform_word(desc_text, type: str, liste_perso:bool=True):
    """ Fonctions de préparation du texte pour le bag of words (Countvectorize et Tf_idf, Word2Vec) 

    cours associé : https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4855001-representez-votre-corpus-en-bag-of-words

    sources externe : 
    - https://inside-machinelearning.com/preprocessing-nlp-tutoriel-pour-nettoyer-rapidement-un-texte/
    - https://www.actuia.com/contribution/victorbigand/tutoriel-tal-pour-les-debutants-classification-de-texte/ 

    La manière la plus simple de représenter un document, c'est ce qu'on a effectué dans le chapitre précédent où l'on a considéré tous les mots utilisés pour chaque artiste, sans distinction ni dépendance par vers, chanson, etc. 
    L'analogie est donc qu'on a considéré chaque artiste par la représentation brute d'un "sac" de tous les mots qu'il a utilisé, sans soucis de contexte (ordre, utilisation, etc).

    # Tf_idf :

    Depuis le départ, on a seulement utilisé les fréquences d'apparition des différents mots/n-grammes présents dans notre corpus. 
    Le problème est que si l'on veut vraiment représenter un document par les n-grammes qu'il contient, il faudrait le faire relativement à leur apparition dans les autres documents.

    En effet, si un mot apparait dans d'autres documents, il est donc moins représentatif du document qu'un mot qui n'apparait que uniquement dans ce document.

    Nous avons d'abord supprimé les mots les plus fréquents de manière générale dans le langage (les fameux stopwords).
    À présent, il ne faut pas considérer le poids d'un mot dans un document comme sa fréquence d'apparition uniquement, mais pondérer cette fréquence par un indicateur si ce mot est commun ou rare dans tous les documents.

    Pour résumer, le poids du n-gramme est le suivant :

    #### poids=fréquence du terme×indicateur similarité

    En l’occurence, la métrique tf-idf (Term-Frequency - Inverse Document Frequency) utilise comme indicateur de similarité l'inverse document frequency qui est l'inverse de la proportion de document qui contient le terme, à l'échelle logarithmique. 
    Il est appelé logiquement « inverse document frequency » (idf). 

    Parameters
    ----------

    desc_text :class:`Series` : le "x"

    type :class:`Choix` : 
        - "classique" = stopword + lower -> Utilisable pour bag of word (countvectorize/tf_idf/word2vec)
        - "lem"  = stopword + lower puis lemmatizer pour bag of word
        - "dl" = lower pour deeplearning

    Return
    ----------

    Series sans stopword puis lower()

    Exemples
    ---------
    >>> data_T['sentence_bow'] = data_T0['text'].apply(lambda x : transform_word(x, "classique"))
    """
    def tokenizer_fct(sentence):  # découpe l'expression en liste de mot
        """
        Découpe l'expression en liste de mot.

        La tokenization désigne le découpage en mots des différents documents qui constituent votre corpus.

        Il existe plusieurs types de tokenisations (mots, phrase...), ici on utilise celui des mots.

        Src : https://www.actuia.com/contribution/victorbigand/tutoriel-tal-pour-les-debutants-classification-de-texte/

        Parameters
        ----------

        sentence : :class:`str` : Phrase à découper

        Return
        ----------
        Liste des mots découpés
        """

        sentence_clean = sentence.replace(
                '-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
        word_tokens = word_tokenize(sentence_clean)
        return word_tokens
    
    # Stop words # génère des mots "stop", comme les pronoms, des verbes d'état, des ponctuations... Des mots dont on ne veut pas
    stop_w = list(set(stopwords.words('english'))) + \
        ['[', ']', ',', '.', ':', '?', '(', ')']
        
    
    # liste personnalisée
    
    stop_perso = ['shipping', 'flipkart'] # à modifier
    
    def stop_word_filter_fct(list_words, liste_perso:bool=True):
        """
        La normalisation et la construction du dictionnaire qui permet de ne pas prendre en compte des détails importants au niveau local (ponctuation, majuscules, conjugaison, etc.)
        Ici on supprime tous les mots qui n'ont aucun intérêt, comme les déterminants et les pronoms

        Source : https://www.actuia.com/contribution/victorbigand/tutoriel-tal-pour-les-debutants-classification-de-texte/

        Parameters
        -----------

        list_words : :class:`list` : liste des mots où il faut exclure les parasites

        Exemples
        ----------
        stop_word = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']

        Return
        ---------
        Liste des mots sans les stopwords
        """
        filtered_w = [
            w for w in list_words if not w in stop_w]  # affiche le mot de la list word s'il n'est pas dans la liste stop_w (mot exact)
        
        if liste_perso is True:
            filtered_w2 = [word for word in filtered_w if not any(wordl in word for wordl in stop_perso)] # si le mot contient une partie de la liste. 
            # par exemple, "flipkart.com" est exclu car contenu dans "flipkart"
            filtered_w3 = [w for w in filtered_w2 if len(w) > 2] #affiche le mot de la liste filtered_w s'il fait plus de 2 lettres
        else:
            filtered_w3 = [w for w in filtered_w if len(w) > 2]
            
        return filtered_w3
    
    def lemma_fct(list_words):  # Lemmatizer (base d'un mot)  # découpe l'expression en liste de lettre
        """
        Le processus de « lemmatisation » consiste à représenter les mots (ou « lemmes » 😉) sous leur forme canonique. 
        Par exemple pour un verbe, ce sera son infinitif. Pour un nom, son masculin singulier. L'idée étant encore une fois de ne conserver que le sens des mots utilisés dans le corpus.
        Dans le processus de lemmatisation, on transforme donc « suis » en « être»  et « attentifs » en « attentif ».

        Source : https://www.actuia.com/contribution/victorbigand/tutoriel-tal-pour-les-debutants-classification-de-texte/

        Parameters
        ----------

        list_words : :class:`list` : liste des mots (de préférence après suppression des stopwords)

        Return
        ----------
        Liste des mots
        """
        lemmatizer = WordNetLemmatizer()
        lem_w = [lemmatizer.lemmatize(w) for w in list_words]
        return lem_w


# lower les mots s'ils ne commencent pas par @ ou http
    def lower_start_fct(list_words):
        """
        Lower les mots s'ils ne commencent pas par @ ou http

        Parameters
        ----------

        list_words : :class:`list` : liste des mots (de préférence après suppression des stopwords)


        Return
        ---------
        Liste des mots après lower()
        """
        lw = [w.lower() for w in list_words if (not w.startswith("@"))
            #                                   and (not w.startswith("#"))
            and (not w.startswith("http"))]
        return lw
    
    
    sentence = tokenizer_fct(desc_text)  # découpage des mots
    sentence = lower_start_fct(sentence)  # lower des mots

    if type in ["classique", "lem"]:
        sentence = stop_word_filter_fct(sentence, liste_perso=liste_perso)  # filtre des stopword

    if type == "lem":
        sentence = lemma_fct(sentence)  # lemmatizer
    # on supprime la liste et on sépare les mots
    transf_desc_text = ' '.join(sentence)
    return transf_desc_text  # on return la phrase traitée


from wordcloud import WordCloud

def visualisation_bag_of_words_general(data, colonne):
    """
    Parameters
    -----------
    
    data = DataFrame
    colonne = colonne Lemmatizer
    
    Return
    ----------
    Visualisation"""
    long_string = ",".join(list(data[colonne].values))
    wordcloud = WordCloud(background_color="white", max_words=5000,
                        contour_width=5, contour_color='steelblue', width=700, height=500)
    # Generate a word cloud
    wc = wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()
    
from IPython.display import display

def visualisation_bag_of_words_categorie(data, colonne, colonne_categorie, categorie):
    """ 
    Parameters
    --------------
    
    data = DataFrame
    colonne = colonne Lemmatizer
    colonne_categorie = colonne avec les categories
    categorie = categorie à filtrer
    
    Return
    ---------------
    Bag of word pour chaque catégorie
    
    Exemple
    --------------

    >>> for categorie in data_txt['categorie'].unique():
    >>> visualisation_bag_of_words_categorie(data_txt, 'description_clean_lem', 'categorie', categorie)"""
    # On filtre sur la catégorie
    data = data[data[colonne_categorie] == categorie]

    long_string = ",".join(list(data[colonne].values))
    wordcloud = WordCloud(background_color="white", max_words=5000,
                          contour_width=5, contour_color='steelblue', width=700, height=500)
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    print(f'Bag of word pour {categorie}')
    display(wordcloud.to_image())

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def creation_bag_of_words(data, feature):
    """ 
    Description : Prépare deux transformateurs : CountVectorizer et Tf-idf

    ## CountVectorizer :

    Comptage de mots

    ## Tf-idf

    TF = nombre de fois où le mot est dans le document / nombre de mots dans le document

    IDF = nombre de documents / nombre de documents où apparaît le mot

    La matrice TF-IDF est définie pour chaque mot relativement à un corpus, comme le produit TF * IDF

    Source : https://www.actuia.com/contribution/victorbigand/tutoriel-tal-pour-les-debutants-classification-de-texte/

    Parameters
    -----------

    data : :class:`DataFrame`
    feature : :class:`Données Lemmatizer`

    Return
    -----------

    cv_transform = le modèle CountVectorizer transformé

    ctf_transform = le modèle Tf-idf transformé

    """
    cvect = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
    ctf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)

    feat = feature
    cv_fit = cvect.fit(data[feat])
    ctf_fit = ctf.fit(data[feat])

    cv_transform = cvect.transform(data[feat])
    ctf_transform = ctf.transform(data[feat])

    print('CV : ')
    print(cvect.get_feature_names_out())
    print('CTF : ')
    print(ctf.get_feature_names_out())

    return cv_transform, ctf_transform



# -----------------------------------

def feature_BERT_fct(model, model_type, sentences, max_length=64, b_size=10, mode='HF'):
    """ Créer les features (BERT)
    
    Source : https://lesdieuxducode.com/blog/2019/4/bert--le-transformer-model-qui-sentraine-et-qui-represente

    Parameters
    -----------

    model : >>> TFAutoModel.from_pretrained(model_type)
    model_type : voir la doc >>> model_type = 'bert-base-uncased' / 'cardiffnlp/twitter-roberta-base-sentiment' "(Modèle pré-entraîné sur des tweets pour l'analyse de sentiment)"
    sentences = :class:`list` : Series utilisable pour le DL, converti avec .to_list()
    max_length = Taille maximale (64 par exemple)
    b_size : batch_size (10 par exemple)
    mode = "HF" par défaut mais "d'autres peuvent être téléchargés comme tfhub."

    Pour le télécharger:
    >>> model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
    >>> bert_layer = hub.KerasLayer(model_url, trainable=True)
    
    import pytorch pour utiliser le gpu


    Return
    -----------
    Features
    
    Last_hidden_stats_tot

    """
    def preprocess_bert(sentences, bert_tokenizer, max_length):
        """
        Préparation des sentences pour BERT

        Source : https://huggingface.co/docs/transformers/v4.20.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus

        import pytorch pour utiliser le gpu

        Parameters
        ----------
        sentences = :class:`list` : Series utilisable pour le DL, converti avec .to_list()
        bert_tokenizer : Exemple >>> bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
        max_length = Taille maximale (64 par exemple)


        Return
        ----------

        input_ids = List of token ids to be fed to a model.
        token_type_ids = List of token type ids to be fed to a model
        attention_mask = List of indices specifying which tokens should be attended to by the model
        bert_inp_tot = contient les trois

        """
        input_ids = []
        token_type_ids = []
        attention_mask = []
        bert_inp_tot = []

        for sent in sentences:
            bert_inp = bert_tokenizer.encode_plus(sent,
                                                  # Attention_mask =  List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens
                                                  add_special_tokens=True,
                                                  max_length=max_length,
                                                  padding='max_length',
                                                  return_attention_mask=True,
                                                  return_token_type_ids=True,
                                                  truncation=True,
                                                  return_tensors="tf")

            input_ids.append(bert_inp['input_ids'][0])
            token_type_ids.append(bert_inp['token_type_ids'][0])
            attention_mask.append(bert_inp['attention_mask'][0])
            bert_inp_tot.append((bert_inp['input_ids'][0],
                                bert_inp['token_type_ids'][0],
                                bert_inp['attention_mask'][0]))

        input_ids = np.asarray(input_ids)
        token_type_ids = np.asarray(token_type_ids)
        attention_mask = np.array(attention_mask)

        return input_ids, token_type_ids, attention_mask, bert_inp_tot

    batch_size = b_size
    batch_size_pred = b_size
    # source : https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
    time1 = time.time()

    for step in range(len(sentences)//batch_size):
        idx = step*batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = preprocess_bert(sentences[idx:idx+batch_size],
                                                                                  bert_tokenizer, max_length)

        if mode == 'HF':    # Bert HuggingFace
            outputs = model.predict(
                [input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode == 'TFhub':  # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids": input_ids,
                                 "input_mask": attention_mask,
                                 "input_type_ids": token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']

        if step == 0:
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else:
            last_hidden_states_tot = np.concatenate(
                (last_hidden_states_tot, last_hidden_states))

    features_bert = np.array(last_hidden_states_tot).mean(axis=1)

    time2 = np.round(time.time() - time1, 0)
    print("temps traitement : ", time2)

    return features_bert, last_hidden_states_tot


import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def feature_USE_fct(sentences, b_size=10):
    """ Créer les features pour USE

    Parameters
    -----------

    sentences = :class:`list` : Series utilisable pour le DL, converti avec .to_list()
    b_size est par défault 10

    Return
    ----------

    Features pour USE
    """
    batch_size = b_size
    time1 = time.time()


    for step in range(len(sentences)//batch_size):
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])


        if step == 0:
            features = feat
        else:
            features = np.concatenate((features, feat))

            

    time2 = np.round(time.time() - time1, 0)
    return features


from pprint import pprint
import pyLDAvis.gensim_models
import pickle
import pyLDAvis


class LDA():
    def __init__(self, data, colonne):
        """ 
        Parameters
        ----------

        data :class:`DataFrame`

        colonne : :class:`Series`: Données lemmatizer
        
        Return
        ----------
        
        sentences = liste de lower mots
        """
        self.data = data
        self.colonne = colonne

        # Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long
        self.sentences = self.data[self.colonne].to_list()
        self.sentences = [gensim.utils.simple_preprocess(
            text) for text in self.sentences]

    def modele_lda(self):
        """
        Créer un dictionnaire avec les mots données.

        Convert `document` into the bag-of-words (BoW) format = list of `(token_id, token_count)` tuples.

        Parameters
        ------------
        sentences :class:`list` Series d'un dataframe où il a été appliqué cette fonction :
        >>> sentences = data[colonne].to_list()
        >>> sentences = [gensim.utils.simple_preprocess(text) for text in sentences]

        Return
        -------------
        id2word = Dictionnaire de mots

        corpus = Liste où les mots ont été convertis en `(token_id, token_count)` tuples
        """
        self.id2word = gensim.corpora.Dictionary(self.sentences)
        self.corpus = [self.id2word.doc2bow(
            sentence) for sentence in self.sentences]
        
        console = Console()
        
        console.print('Création du dictionnaire et des tokens', style="green")

    def training_lda(self, num_topics=10, iterations=500, passes=1):
        """ Entraine le modèle LDA

        Parameters
        -----------

        Corpus :class:`list` Liste de mots sous la forme `(token_id, token_count)` tuples

        id2word :class:`list` Dictionnaire de mots

        num_topics :class:`ìnt` Nombre de topics

        Return
        -----------
        lda_model = Modèle de liste des topics
        >>> pprint(self.lda_model.print_topics())

        """
        # LDA building
        self.lda_model = gensim.models.LdaMulticore(corpus=self.corpus,
                                                    id2word=self.id2word,
                                                    num_topics=num_topics,
                                                    iterations=iterations,
                                                    passes=passes)
        # Print the Keyword in the 10 topics
        pprint(self.lda_model.print_topics())

    def visualisation_lda(self, save: bool = False):
        """ Permet de visualiser les mots clés de chaque topic

        Parameters
        -----------

        lda_model :class:`LDAMulticore` modèle entrainé (voir training_lda)

        Corpus :class:`list` Liste de mots sous la forme `(token_id, token_count)` tuples

        id2word :class:`list` Dictionnaire de mots

        Return
        -----------

        visualisation = Graphique html


        """
        # Visualize the topics
        pyLDAvis.enable_notebook()
        self.visualisation = pyLDAvis.gensim_models.prepare(
            self.lda_model, self.corpus, self.id2word)

        if save is True:
            pyLDAvis.save_html(self.visualisation, './visualisation.html')
            
            
class Word2vec():


    def __init__(self, data, colonne, w2v_size=300, w2v_window=5, w2v_min_count=1, w2v_epochs=100, maxlen=64):
        """
        Parameters
        ----------

        data :class:`DataFrame`

        colonne : :class:`Series`: Données lemmatizer
        """
        self.data = data
        self.colonne = colonne
        self.w2v_size = w2v_size  # Dimensionality of the word vectors.
        # Maximum distance between the current and predicted word within a sentence.
        self.w2v_window = w2v_window
        # Ignores all words with total frequency lower than this.
        self.w2v_min_count = w2v_min_count
        # Number of iterations (epochs) over the corpus.
        self.w2v_epochs = w2v_epochs
        self.maxlen = maxlen

        # Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long
        self.sentences = self.data[self.colonne].to_list()
        self.sentences = [gensim.utils.simple_preprocess(
            text) for text in self.sentences]

    # Création et entraînement du modèle Word2Vec

    def modele_word2vec(self):
        """
        Création et entrainement du modèle Word2Vec

        Parameters
        ----------

        Sentences :class:`list` : Données lemmatizer
        w2v_min_count : :class:ìnt` : Ignores all words with total frequency lower than this.

        Return
        ----------

        w2v_model = Train, use and evaluate neural networks described in https://code.google.com/p/word2vec/
        model_vectors = Dictionnaire avec les vecteurs de chaque mot.
        >>> model_vectors.key_to_index pour avoir la position de chaque mot
        w2v_words = Liste des mots
        """
        print("Build & train Word2Vec model ...")
        self.w2v_model = gensim.models.Word2Vec(min_count=self.w2v_min_count, window=self.w2v_window,
                                                vector_size=self.w2v_size,
                                                seed=42,
                                                workers=1)
        #                                                workers=multiprocessing.cpu_count())
        # prépare le vocabulaire du modèle
        self.w2v_model.build_vocab(self.sentences)
        self.w2v_model.train(
            self.sentences, total_examples=self.w2v_model.corpus_count, epochs=self.w2v_epochs)
        self.model_vectors = self.w2v_model.wv
        self.w2v_words = self.model_vectors.index_to_key
        print("Vocabulary size: %i" % len(self.w2v_words))
        print("Word2Vec trained")

    def transformation_sentences_integers(self):
        """ Transforme les phrases en une séquence de integer

        Parameters

        sentences : :class:`list` Données lemmatizer
        maxlen : max length

        Return
        ----------

        tokenizer = le tokenizer
        x_sentences = Liste integer

        """

        print("Fit Tokenizer ...")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.sentences)
        # texts to sequence fait la transformation des strings en int
        # pad_sequences transforme plusieurs séquences en une double liste
        self.x_sentences = pad_sequences(self.tokenizer.texts_to_sequences(self.sentences),
                                         maxlen=self.maxlen,
                                         padding='post')

        self.num_words = len(self.tokenizer.word_index) + 1
        print("Number of unique words: %i" % self.num_words)

    def matrice(self):
        """ Cours associé : https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4855006-effectuez-des-plongements-de-mots-word-embeddings

        Parameters
        -----------

        tokenizer :class:`tokenizer`

        w2v_words et model_vectors  > voir modele_word2vec 

        Return
        -----------

        embedding_matrix = Prend les résultats de chaque mot du dictionnaire model_vectors sans None 
        vocab_size = Nombre de mots   
        """
        print("Create Embedding matrix ...")
        self.word_index = self.tokenizer.word_index  # ordre des mots
        self.vocab_size = len(self.word_index) + 1  # nombre de mots
        self.embedding_matrix = np.zeros((self.vocab_size, self.w2v_size))
        i = 0
        j = 0

        for word, idx in self.word_index.items():
            i += 1
            if word in self.w2v_words:
                j += 1
                self.embedding_vector = self.model_vectors[word]
                if self.embedding_vector is not None:
                    self.embedding_matrix[idx] = self.model_vectors[word]

        self.word_rate = np.round(j/i, 4)
        print("Word embedding rate : ", self.word_rate)
        print("Embedding matrix: %s" % str(self.embedding_matrix.shape))
        
        console = Console()
        
        console.print('Fait !', style="green")

    def embedding(self):
        input = Input(shape=(len(self.x_sentences),
                      self.maxlen), dtype='float64')
        self.word_input = Input(shape=(self.maxlen,), dtype='float64')
        self.word_embedding = Embedding(input_dim=self.vocab_size,
                                        output_dim=self.w2v_size,
                                        weights=[self.embedding_matrix],
                                        input_length=self.maxlen)(self.word_input)
        self.word_vec = GlobalAveragePooling1D()(self.word_embedding)
        self.embed_model = Model([self.word_input], self.word_vec)

        print(self.embed_model.summary())

    def predict(self):
        """Return
        -----------
        embeddings = modele final avec les features"""
        self.embeddings = self.embed_model.predict(self.x_sentences)
        print(self.embeddings.shape)
        console = Console()
        
        console.print('Modèle terminé !', style="green")
