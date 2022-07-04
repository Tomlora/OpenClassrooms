import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.decomposition import PCA
from sklearn import preprocessing
from rich.console import Console

class ACP_kevin():
    def __init__(self, colonne=None, dataframe=None):
        """ Class ACP
        
        Parameters
        ------------
        
        Colonne du dataframe avec les données à ACP
        
        Return
        ------------
        
        data = données 
        
        n = nombre d'observations
        
        p = nombre de variables
        
        std_scale = fit StandardScaler
        
        x_scaled = transform StandardScaler
        
        """
        if colonne != None:
            self.data = colonne
        else:
            self.data = dataframe[colonne]
        self.n = self.data.shape[0]
        self.p = self.data.shape[1]
        self.std_scale = preprocessing.StandardScaler().fit(self.data.values)
        self.X_scaled = self.std_scale.transform(self.data.values)
        
    def initialisation_acp(self):
        """ Initialise l'acp
        
        Parameters
        -----------
        
        X_scaled : Données scalées
        
        Return
        -----------
        
        acp = l'acp
        
        coord = données scalées puis acp
        
        n_comp = nombre de composants générées
        
        eigval = variance préservée par l'acp à chaque composant
        
        
        """
        self.acp = PCA(svd_solver='full')
        self.coord = self.acp.fit_transform(self.X_scaled)
        self.n_comp = self.acp.n_components_
        self.eigval = self.acp.explained_variance_ratio_
        
        console = Console()
        
        console.print(f'ACP initialisé avec {self.n_comp} composants ', style="green")
        
        # calcul composantes principales
        
    def optimisation_acp(self, n_comp = None):
        """ avec le nombre de composantes retenues via scree_plot_acp
        
        Parameters
        -------------
        
        n_comp (par défault, le maximum)
        
        Return
        -------------
        
        pca_opti = pca avec nombre de composants retenus
        
        pcs = Principal axes in feature space, representing the directions of maximum variance in the data.
    
        X_projected = valeurs des composants
        
        """
        if n_comp == None:
            self.n_comp_opti = self.n_comp
        
        self.pca_opti = PCA(n_components=self.n_comp_opti)
        self.pca_opti.fit(self.X_scaled)
        self.pcs = self.pca_opti.components_
        self.X_projected = self.pca_opti.transform(self.X_scaled)
        
        console = Console()
        
        console.print(f'ACP optimisé', style="green")
            
   
        

    def scree_plot_acp(self):
        ''' Détermine le meilleur nombre de composants à prendre
        
        Parameters
        ----------
        acp : :class:`acp` l'acp initialisé, après avoir fait un .fit_transform
        p : nombre de variables qui ont été scalés(>>> p = df.shape[1])
        
        Return
        -----------
        
        Deux graphiques :
        - Un graphique avec la variance de chaque composant
        - Un graphique avec la variance cumulée de chaque composant
        '''
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

        #Graphique 1

        axes[0].plot(np.arange(1,self.p+1),self.eigval)
        axes[0].set_title("Scree plot")
        axes[0].set_ylabel("Eigen values")
        axes[0].set_xlabel("Factor number")

        # Graphique 2

        axes[1].plot(np.arange(1,self.p+1),np.cumsum(self.eigval))
        axes[1].set_title("Cumul Scree plot")
        axes[1].set_ylabel("Cumsum explained variance ratio")
        axes[1].set_xlabel("Factor number")
        plt.show()
        
        
    def cercle_correlation_acp(self, axis_ranks, labels=None, label_rotation=0, lims=None):
        '''
        Parameters
        ----------
        
        self.pcs : :class:`pca_components_`
        n_comp : :class:`nombre de composants`
        pca : :class:`pca` après avoir utilisé decomposition.PCA et pca.fit()
        
        axis_rank : :class:`Les axes` Notation : `([0,1])` ou `[(0,1),(2,3),(4,5)]`
        
        labels : :class:`list`: les noms des variables à représenter.
        
        Exemples
        --------
        >>> pca = decomposition.PCA(n_components=n_comp)
        >>> pca.fit(X_scaled)

        >>> self.pcs = pca.components_
        >>> cercle_correlation_acp(self.pcs, n_comp, pca, [(0,1)], labels=colonnes_nutrition_KNN_colonnes)
        '''
        for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
            if d2 < self.n_comp_opti:
                # initialisation de la figure
                fig, ax = plt.subplots(figsize=(10,9))
                # détermination des limites du graphique
                if lims is not None :
                    xmin, xmax, ymin, ymax = lims
                elif self.pcs.shape[1] < 30 : #si le nombre de variables est inférieur à 30
                    xmin, xmax, ymin, ymax = -1, 1, -1, 1
                else :
                    xmin, xmax, ymin, ymax = min(self.pcs[d1,:]), max(self.pcs[d1,:]), min(self.pcs[d2,:]), max(self.pcs[d2,:])
                # affichage des flèches
                lines = [[[0,0],[x,y]] for x,y in self.pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
                
                # affichage des noms des variables  
                if labels is not None:  
                    for i,(x, y) in enumerate(self.pcs[[d1,d2]].T):
                        if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                            plt.text(x, y, labels[i], 
                                    fontsize='14', 
                                    ha='center', 
                                    va='center', 
                                    rotation=label_rotation, 
                                    color="blue", alpha=0.5)
                
                # affichage du cercle
                circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)
                

                # définition des limites du graphique
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
            
                # affichage des lignes horizontales et verticales
                plt.plot([-1, 1], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-1, 1], color='grey', ls='--')

                # nom des axes, avec le pourcentage d'inertie expliqué
                plt.xlabel('F{} ({}%)'.format(d1+1, round(100*self.pca_opti.explained_variance_ratio_[d1],1)))
                plt.ylabel('F{} ({}%)'.format(d2+1, round(100*self.pca_opti.explained_variance_ratio_[d2],1)))

                plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
                plt.show(block=False) #If False ensure that all figure windows are displayed and return immediately. In this case,
                #you are responsible for ensuring that the event loop is running to have responsive figures.

        
    def graphique_acp(self, axis_ranks, labels=None, alpha=1, illustrative_var=None):
        '''
        Parameters
        ----------
        X_projected : :class:`pca.transform(X_scaled)`
        n_comp : :class:`nombre de composants`
        pca : :class:`pca` après avoir utilisé decomposition.PCA et pca.fit()
        
        axis_ranks : :class:`Les axes` Notation : `([0,1])` ou `[(0,1),(2,3),(4,5)]`
        
        labels : :class:`list`: les noms des variables à représenter.
        
        Exemples
        --------
        >>> X_projected = pca.transform(X_scaled)
        >>> graphique_acp(X_projected, n_comp, pca, [(0,1)], labels = np.array(data_final_KNN['product_name']), illustrative_var=data_final_KNN['nutrition_grade_fr'])
        
        
        
        '''
        for d1,d2 in axis_ranks:
            if d2 < self.n_comp_opti:
    
                # initialisation de la figure       
                fig = plt.figure(figsize=(7,6))
            
                # affichage des points
                if illustrative_var is None:
                    plt.scatter(self.X_projected[:, d1], self.X_projected[:, d2], alpha=alpha)
                else:
                    illustrative_var = np.array(illustrative_var)
                    for value in np.unique(illustrative_var):
                        selected = np.where(illustrative_var == value)
                        plt.scatter(self.X_projected[selected, d1], self.X_projected[selected, d2], alpha=alpha, label=value)
                    plt.legend()


def anova(X, Y, data):
    '''
    Permet d'étudier l'effet des variables qualitatives sur une variable quantitative.
    L’ANOVA sert concrètement à mettre en lumière l’existence d’une interaction entre ces facteurs de variabilité et la variable quantitative principale étudiée
    
    X : Variable qualitative à expliquer
    Y : Variables quantitatives
    data : :class:`DataFrame` comportant X/Y
    
    Pré-requis
    ----------
    En cas d'analyse poussée, il faut faire un test de Fisher (indépendance distribution, variance)
    
    Return
    ---------
    DataFrame avec : 
    Source = Factor names

    SS = Sums of Squares

    DF = Degrees of freedom

    MS = Mean Squares

    F = F-values

    p-unc : uncorrected p-values

    np2 : Eta_square
    
    Source
    --------
    https://pingouin-stats.org/generated/pingouin.anova.html
    
    https://spss.espaceweb.usherbrooke.ca/analyse-de-variance/
    '''
    
    df_aov = pd.DataFrame(columns=['X', 'Y', 'SS', 'DF', 'MS', 'F', 'p-unc', 'np2'])
    # test de Fisher (3 hypothèses : indépendance, distribution, variance (même hauteur sur boxplot))

    for y in Y:
        aov = pg.anova(data=data, dv=y,between=X,detailed=True)
        aov.rename(columns={'Source' : 'X'}, inplace=True )
        aov.insert(1, 'Y', y)
        df_aov = df_aov.append(aov)

        
    df_aov = df_aov.reset_index()
        
    df_aov
