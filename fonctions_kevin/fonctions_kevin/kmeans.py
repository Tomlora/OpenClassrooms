import pandas as pd
import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt

from tqdm import tqdm
import matplotlib.cm as cm
import plotly.express as px
from IPython.display import display  # permet de montrer un dataframe plutôt qu'un "tableau pas très joli" avec print

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score





############################ Kmeans

def elbow_methode(scaled_features) -> plt:
    """
    Choisir le meilleur cluster. 
    
    Parameters
    ----------
    scaled_features : :class:`Données scalées avec un scaler`
    
    Return
    ----------
    Graphique
    
    Choix du cluster
    ----------
    Prendre celui où la courbe s'adoucit.
    
    Source
    ---------
    https://medium.com/@sk.shravan00/k-means-for-3-variables-260d20849730#:~:text=You%20need%20to%20consider%203,values%20from%20the%20data%20set.
    """
    sse = []
    for k in range(1, 8):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
        
    plt.plot(range(1, 8), sse)
    plt.title('Elbow methode')
    plt.xlabel("Nombre de clusters")
    plt.ylabel("SSE")
    plt.show()
    
    
def do_silhouette_score(scaled_features) -> plt:
    """Pour chaque point, son coefficient de silhouette est la différence entre la distance moyenne avec les points du même groupe que lui (cohésion)
     
    et la distance moyenne avec les points des autres groupes voisins (séparation).
    
    Parameters
    ----------
    scaled_features : :class:`Données scalées avec un scaler`
    
    Il faut au minimum 2 clusters pour le silhouette score.
    
    Return
    ----------
    Graphique
    
    Choix du cluster
    ----------
    La plus haute valeur
    """
    silhouette_coefficients = []



    for k in tqdm(range(2, 8)):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.plot(range(2,8), silhouette_coefficients)
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Silhouette coefficients")
    plt.show()
    
    
def silhouette_analyse(scaled_features):
    """
    Silhouette analysis can be used to study the separation distance between the resulting clusters. 
    
    The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually. 
    
    This measure has a range of [-1, 1].

    Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters. 
    
    A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.
    
    Parameters
    ----------
    scaled_features : :class:`Données scalées avec un scaler`
    
    Return
    ----------
    DataFrame et Graphique
    
    Choix du cluster
    ---------
    Plus le silhouette avg est bon, mieux c'est.
    Toutefois, il faut faire attention : Dans le graphique de gauche, il faut que tous les clusters aient dépassé la ligne rouge. Sinon le choix est mauvais.
    
    Source
    ---------
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    """
    liste_clusters = []
    average_score = []
    range_n_clusters = np.arange(2,8,1)

    for n_clusters in tqdm(range_n_clusters):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(scaled_features) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(scaled_features)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        liste_clusters.append(n_clusters)
        average_score.append(silhouette_avg)
        
        

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(scaled_features, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            scaled_features[:, 0], scaled_features[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )
        
        # When you use enumerate(), the function gives you back two loop variables:

        # The count of the current iteration
        # The value of the item at the current iteration
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
    
    df_silhouette_score = pd.DataFrame([liste_clusters, average_score]).transpose()
    df_silhouette_score = df_silhouette_score.rename(columns={0: 'n_clusters', 1:'silhouette_score_avg'})
    df_silhouette_score.set_index('n_clusters', inplace=True)
    display(df_silhouette_score)

    plt.show()
    

    
def k_means_line_polar(df_line_polar, log:bool, algo, value="", colonne="cluster") -> px:
    ''' Radar sur le profil de segmentation
    
    Parameters
    ----------
    df_line_polar : :class: `DF avec les colonnes de segmentation`
    
    log : :class: `bool` -> Faut-il log les valeurs ?
    
    algo : :class: `K-means`
    
    value : :class: `Optionnel`-> si les valeurs diffèrent du df
    
    colonne : :class: `str` -> Colonne du DF avec les clusters
    
    '''
    if value == "": # si pas de valeur, on prend les valeurs du DF
        clusters=pd.DataFrame(df_line_polar,columns=df_line_polar.columns)
    else:
        clusters=pd.DataFrame(value,columns=df_line_polar.columns)
    clusters[colonne]=algo.labels_
    polar=clusters.groupby(colonne).mean().reset_index()
    polar=pd.melt(polar,id_vars=[colonne])
    if log is True:
        polar['value'] = np.log(polar['value'])
    else:
        polar['value'] = polar['value']
    fig = px.line_polar(polar, r="value", theta="variable", color=colonne, line_close=True,height=800,width=1400)
    fig.show()