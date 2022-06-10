import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler



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
