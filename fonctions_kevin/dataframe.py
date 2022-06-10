import ast 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import plotly.express as px




########################## - DataFrame

def cellules_manquantes_pourcentage(data:pd.DataFrame):
    """Permet d'avoir un % de cellules manquantes

    Parameters
    -----------
    data: :class:`DataFrame`
            DataFrame avec la colonne spécifiée
    
    Exemples:        
    ----------
    pourcentage_valeur_manquantes = cellules_manquantes_pourcentage(data[column])
    """
    
    # Permet d'avoir un % de cellules manquantes
    return data.isna().sum().sum()/(data.size)


def extraire_variables_imbriquees(df:pd.DataFrame, colonne:str):
    """Permet de développer les dictionnaires retenues dans une variable pour en faire des variables indépendantes
    

    Parameters
    -----------
    df: :class:`DataFrame`
            DataFrame
    colonne: :class:`str`
            Le nom de la colonne
    
    Return
    -----------
    Le nouveau DataFrame
    """

    # Vocabulaire à connaitre : liste/dictionnaire en compréhension
    df[colonne] = [ast.literal_eval(str(item)) for index, item in df[colonne].iteritems()]

    df = pd.concat([df.drop([colonne], axis=1), df[colonne].apply(pd.Series)], axis=1)
    return df

def supprimer_variables_peu_utilisables(data:pd.DataFrame, condition:float):
    """Supprime les variables qui ont un nombre de données manquantes important. 
    
    On calcule en fonction d'un %.
    
    Fonctions "cellules_manquantes_pourcentage" requise !

    Parameters
    -----------
    data: :class:`DataFrame`
            Le DataFrame
    condition: :class:`float`
            % de données manquantes. Au-dessus de ce nombre, on supprime.
    
    Return
    ----------
    Le DataFrame modifié
    """
    # Permet de supprimer les variables qui ont un nombre de données manquantes important. On calcule en fonction d'un % (condition)
    data_filtree = pd.DataFrame()
    data_filtree = data
    for column in data.columns:  # on boucle sur chaque colonne de la Dataframe
        var_type = data[column].dtypes  # on check le type de la colonne
        pourcentage_valeur_manquantes = cellules_manquantes_pourcentage(data[column])   # % de données manquant
        if var_type == 'float64' and float(pourcentage_valeur_manquantes) > condition:  # si le type n'est pas float, ça ne peut pas marcher.
            data_filtree.drop(column, axis=1, inplace=True)  # on drop l'intégralité de la colonne si le % manquant dépasse la condition...
            print(f'La colonne {column} , avec {pourcentage_valeur_manquantes*100} % de données manquantes de la Dataframe est supprimée')
    return data_filtree


def valeur_unique(df:pd.DataFrame):
    """Permet de voir les valeurs uniques.
    
    Si moins de 20 valeurs, on les affiche. Sinon, affiche le nombre de valeurs uniques.

    Parameters
    -----------
    df: :class:`DataFrame`
            Le DataFrame
    """
    # Si moins de 20 valeurs uniques, les affiche. Sinon, affiche le nombre de valeurs uniques
    for column in df.columns:
        if df[column].nunique() < 20:
            print('Colonne {}, valeurs uniques :\n{}\n'.format(column, df[column].unique()))
        else:
            print('Colonne {}, {} valeurs uniques'.format(column, df[column].nunique()))
            
            
def comparaisons_colonnes(list_1:list, list_2:list):
    """Compare les colonnes de deux listes, et renvoient celles qui ne sont pas présentes dans les deux fichiers.

    Parameters
    -----------
    list_1: :class:`list`
            Colonnes du DataFrame 1
    list_2: :class:`list`
            Colonnes du DataFrame 2
            
    Return
    ----------
    Deux listes avec la différence dans le DataFrame 1 puis le DataFrame 2
    
    Exemples
    ----------
    list_1 = list(df_2015.columns)
    
    list_2 = list(df_2016.columns)

    dif2015, dif2016 = comparaisons_colonnes(list_1, list_2)
    """
    # Un set est un ensemble de clefs non ordonnées et non redondant où l'on peut savoir
    # si un élément est présent sans avoir à parcourir toute la liste (une sorte de dictionnaire où les valeurs seraient ignorées, seules les clefs comptent).
    dif_list_1_list_2 = list(set(list_1) - set(list_2))
    dif_list_2_list_1 = list(set(list_2) - set(list_1))
    return dif_list_1_list_2, dif_list_2_list_1


########################## - Préparation Dataframe pour prédiction


def df_filtre_categorical(df:pd.DataFrame):
    """Identifie dans un DataFrame les colonnes catégoriques.

    Parameters
    -----------
    df: :class:`DataFrame`
            Le DataFrame
            
    Return
    ----------
    Liste des colonnes catégoriques
    
    Exemple
    ----------
    categorical_columns = df_filtre_categorical(df)
    """
    categorical_columns = df.select_dtypes(['object']).columns
    #retourne le nom des colonnes
    return categorical_columns

def df_filtre_numerical(df:pd.DataFrame):
    """Identifie dans un DataFrame les colonnes numériques.

    Parameters
    -----------
    df: :class:`DataFrame`
            Le DataFrame
            
    Return
    ----------
    Liste des colonnes numériques
    
    Exemple
    ----------
    numerical_columns = df_filtre_numerical(df)
    """
    numerical_columns = df.select_dtypes(['int64', 'float64']).columns

    return numerical_columns

def df_encodage_categorie(df:pd.DataFrame, categorical_columns:list):
    """Utilise un OneHotEncoding pour transformer les colonnes catégoriques

    Parameters
    -----------
    df: :class:`DataFrame`
            Le DataFrame
    categorical_columns: :class:`Liste`
            Colonnes catégoriques 
            
    Return
    ----------
    Le OHE encoding et ses paramètres
    
    Les colonnes transformées.
    
    Exemple
    ----------
    ohe, df_ohe = df_encodage_categorie(df, categorical_columns)

    
    Pour les inclure dans le DataFrame de base:
    
        df.reset_index(drop=True, inplace=True)
    
        df[df_ohe.columns.values] = df_ohe
    """

    ohe = OneHotEncoder(sparse=False)  # Will return sparse matrix if set True else will return an array.
    cat_enc = ohe.fit_transform(df[categorical_columns])

    df_ohe = pd.DataFrame(cat_enc)
    
    return ohe, df_ohe

def df_scaling_numeric(df:pd.DataFrame, numerical_columns:list):
    """Standardise et normalise les données numériques avec un standardScaler

    Parameters
    -----------
    df: :class:`DataFrame`
            Le DataFrame
    numerical_columns: :class:`Liste`
            Les colonnes à standardiser
            
    Return
    ----------
    Dataframe avec les colonnes standardisées
    
    Exemple
    ----------
    df[numerical_columns] = df_scaling_numeric(df, numerical_columns)
    """
    ss = StandardScaler()
    df[numerical_columns] = ss.fit_transform(df[numerical_columns])
    
    # exemple : df[numerical_columns] = df_scaling_numeric(df, numerical_columns)

    return df[numerical_columns]

# fusion des colonnes numériques et des colonnes onehotencoders
# liste_train = numerical_columns.tolist() + df_ohe.columns.values.tolist()
# liste_features = list(numerical_columns) + list(ohe.get_feature_names_out(categorical_columns))
