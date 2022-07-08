import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler



########################## - Préparation Dataframe pour prédiction


def df_filtre_categorical_numerique(df:pd.DataFrame):
    """Identifie dans un DataFrame les colonnes catégoriques/numeriques et les séparent

    Parameters
    -----------
    df: :class:`DataFrame`
            Le DataFrame
            
    Return
    ----------
    Liste des colonnes catégoriques
    Liste des colonnes_numeriques
    
    Exemple
    ----------
    categorical_columns, numerical_columns = df_filtre_categorical(df)
    """
    categorical_columns = df.select_dtypes(['object']).columns
    numerical_columns = df.select_dtypes(['int64', 'float64']).columns
    
    #retourne le nom des colonnes
    return categorical_columns, numerical_columns


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
    donnees_scalees = df_scaling_numeric(df, numerical_columns)
    """
    ss = StandardScaler()
    df[numerical_columns] = ss.fit_transform(df[numerical_columns])
    
    return df[numerical_columns]

def fusion_ss_ohe(numerical_columns, df_ohe):
    '''fusion des colonnes numériques et ohe
    
    Si ça ne marche pas, faire: 
    
    >>> liste_features = list(numerical_columns) + list(ohe.get_feature_names_out(categorical_columns))'''
    liste_train = numerical_columns.tolist() + df_ohe.columns.values.tolist() 
    return liste_train



