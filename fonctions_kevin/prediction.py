import ast 
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, validation_curve
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
import sklearn.metrics
from xgboost import XGBRegressor
from sklearn import svm
import matplotlib.pyplot as plt


import plotly.express as px







########################## - Entrainement prédiction

def training_model_regression(df_train:pd.DataFrame, Y_target:pd.DataFrame, test_size:float=0.33, random_state:int=0):
    """Teste plusieurs modèles de regression.
    
    Effectue un scoring / cross-validation / MAE / MSE / RMSE pour les algorithmes suivants :
        - Régression Linéaire
        - Lasso
        - Ridge
        - ElasticNet
        - RandomForestRegressor
        - XGBRegressor
        - SVR
        
    Effectue deux graphiques :
        - Points réels et prédits
        - Validation curve en fonction d'un hyperparamètre

    Parameters
    -----------
    df_train: :class:`DataFrame`
            Dataset d'entrainement : Il faut inclure les colonnes pour l'algorithme . `Données scalées`
    Y_target: :class:`Colonne DataFrame`
            La variable Target
    test_size: :class:`Float`
            Par défault = 0.33
    random_state: :class:`DataFrame`
            Par défault = 0
            
    Return
    ----------
    Dataframe avec un résumé des résultats de chaque algorithme
    
    Exemple
    ----------
    df_ghg = training_model(df[liste_train], Y['Target'], random_state=0)
    ici, liste_train = colonne_numerique (standardisée) + colonnes du ohe : list(numerical_columns) + list(ohe.get_feature_names_out(categorical_columns))
    """
    
    
    #df_ghg = training_model(df[liste_train], Y['TotalGHGEmissions_avg'], random_state=0)
    # Src : https://www.youtube.com/watch?v=w_bLGK4Pteo (cross validation score / validation_curve)
    
    
    # listes qui vont servir à créer un dataframe
    score_training_liste = []
    cross_liste = []
    mae_liste = []
    mse_liste = []
    rmse_liste = []
      
    
    # la variable visée
    
    X_train, X_test, y_train, y_test = train_test_split(df_train,
                                                    Y_target,
                                                    test_size=test_size,
                                                    random_state=random_state) 
    
    # les algos qu'on va tester
    algos = {
    'LinearRegression' : LinearRegression(),
    'Lasso' : Lasso(tol=0.5),
    'Ridge' : Ridge(),
    'ElasticNet' : ElasticNet(),
    'RandomForestRegressor' : RandomForestRegressor(n_jobs=-1),
    'XGBRegressor': XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=-1),
    'SVR' : svm.SVR(cache_size=7000, max_iter=1000)
}
    # les hyper paramètres qu'on va faire varier pour trouver le meilleur score
    hyperparametres = {
    'LinearRegression' : ['n_jobs', np.arange(1,50)],  # The number of jobs to use for the computation
    'Lasso' : ['alpha', np.arange(0.01,1, 0.05)],
    'Ridge' : ['alpha', np.arange(0.01,1, 0.05)],
    'ElasticNet' : ['alpha', np.arange(0.01, 1, 0.05)],
    'RandomForestRegressor' : ['n_estimators', [20,50,100,500,1000,2000]],  #nombre d'arbres dans la foret
    'XGBRegressor': ['n_estimators', [20,50,100,500,1000,2000]], # nombre d'arbres dans la foret
    'SVR' : ['gamma', [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]]
}
    # on teste
    for algo_name, algo in algos.items():
        model = algo.fit(X_train, y_train)
        score_entrainement = model.score(X_test, y_test)
        cross_validation = cross_val_score(algo, X_train, y_train, cv=5).mean() # permet de découper le trainset en 5 parties et de tester. On fait ensuite la moyenne des notes :)
        train_score, val_score = validation_curve(algo, X_train, y_train, param_name=hyperparametres[algo_name][0], param_range=hyperparametres[algo_name][1], cv=5)
        predict = model.predict(X_test)
        
        
    #Pour évaluer un modèle de régression, on peut calculer la distance entre valeurs prédites et vraies valeurs. Cela nous donne :
    #la somme des carrés des résidus (RSS) ;
    # la moyenne de cette somme (MSE) ;
    # la racine carrée de cette moyenne (RMSE).
        mae = sklearn.metrics.mean_absolute_error(y_test, predict) # Mean absolute error regression loss.
        mse = np.square(np.subtract(y_test,predict)).mean() 
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_test, predict)) # Mean squared error regression loss.

        # on ajoute les résultats dans les listes pour le dataframe
        score_training_liste.append(score_entrainement) # note : r2_score donne le mm resultat
        cross_liste.append(cross_validation)
        mae_liste.append(mae)
        mse_liste.append(mse)
        rmse_liste.append(rmse)
        
        # on affiche les résultats

        
        print(algo_name)
        print('Score entrainement: ' + str(score_entrainement))
        print('Cross: ' + str(cross_validation))

        '''
        The mean absolute error measures the average differences between predicted values and actual values
        Unlike the mean squared error (MSE), the MAE calculates the error on the same scale as the data. This means it’s easier to interpret.
        The MAE doesn’t square the differences and is less susceptible to outliers
        Both values are negatively-oriented. This means that, while both range from 0 to infinity, lower values are better.
        For example, in our earlier example of a MAE of 10, if the values ranged from 10,000 to 100,000 a MAE of 10 would be great. However, if the values ranged from 0 through 20, a MAE would be terrible.
        Src : https://datagy.io/mae-python/
        '''
        
        print('MAE : ' + str(mae))
        
        # So, MSE is a risk function that helps us determine the average squared difference between the predicted and the actual value of a feature or variable.
        
        print('MSE : ' + str(mse))
        
        '''
        RMSE is an acronym for Root Mean Square Error, which is the square root of value obtained from Mean Square Error function.

        Using RMSE, we can easily plot a difference between the estimated and actual values of a parameter of the model.

        By this, we can clearly judge the efficiency of the model.

        Usually, a RMSE score of less than 180 is considered a good score for a moderately or well working algorithm. 
        In case, the RMSE value exceeds 180, we need to perform feature selection and hyper parameter tuning on the parameters of the model.
        '''
        
        print('RMSE : ' + str(rmse))
        
        
        # Graphique des résultats
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        ax1.scatter(y_test, predict, c='coral')
        ax1.set_title(f'Prediction : {Y_target.name} avec {algo_name}')
        ax1.set_xlabel("Vraies valeurs", size = 12)
        ax1.set_ylabel("Valeurs prédictes", size = 12)
        
        ax2.plot(hyperparametres[algo_name][1], val_score.mean(axis=1), label="validation")
        ax2.plot(hyperparametres[algo_name][1], train_score.mean(axis=1), label="train")
        ax2.set_title(f'Validation curve ({algo_name})')
        ax2.set_xlabel(hyperparametres[algo_name][0], size = 12)
        ax2.set_ylabel("score", size = 12)
        ax2.legend()
        
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=5, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
        

        plt.show()
        
        print('--------')
        
    # le dataframe
        
    df_liste = pd.DataFrame([algos.keys(), score_training_liste, cross_liste, mae_liste, mse_liste, rmse_liste])
    df_liste = df_liste.transpose()
    df_liste = df_liste.rename(columns={0:'Model', 1:'Score training', 2: 'Cross Validation', 3: 'MAE', 4:'MSE', 5:'RMSE'})
        
    return df_liste


########################## - Graphique Prédiction


def feature_importance(model, classifier:str, index:list):
    """Graphique Feature Importance

    Parameters
    -----------
    model: :class:`Modele de regression`
            Modèle de régression entrainé 
            
            Exemples :
                model = XGBRegressor()
                
                model.fit(X_train, y_train)
    classifier: :class:`str`
            Nom du Modèle pour le titre du graphique
    index: :class:`Liste`
            Liste des variables qui ont été utilisés dans l'algorithme
            
    Return
    ----------
    Graphique
    """
    df_features = pd.DataFrame(model.feature_importances_, index=index)
    fig = px.histogram(x=df_features[0], y=df_features.index, color=df_features.index, title="Features importance " + str(classifier), width=1600, height=1400).update_yaxes(categoryorder="total ascending")
    fig.update_layout(showlegend=False)
    return fig