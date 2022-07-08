import ast 
import pandas as pd





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


