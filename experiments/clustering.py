
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cox_regression_main as cox


# Visualize silhouette coefficients of the data.
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
def generate_silhouette_plot(X, cluster_labels, n_clusters):
    # Create a subplot with 1 row and 2 columns
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1.
    ax.set_xlim([-0.25, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters.
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig("clustering_analysis/silhouette_plot_%sclusters.png" % n_clusters)


def analyze_clustering():
    # Get clinical and gene expression data.
    clinical_tsv = "processed_data/clinical_processed.tsv"
    clinical_df = pd.read_csv(clinical_tsv, sep="\t")

    gexp_top05_tsv = "processed_data/gene_expression_top05_matrix.tsv"
    gexp_df = pd.read_csv(gexp_top05_tsv, sep="\t")

    cox_data = cox.CoxRegressionDataset(gexp_df, clinical_df, standardize=False)
    X_train, y_train = cox_data.X, cox_data.y

    num_cluster_range = [2, 3, 4, 5, 6]
    for k in num_cluster_range:
        clusterer = KMeans(n_clusters=k, random_state=10)
        cluster_labels = clusterer.fit_predict(X_train)

        # Silhouette analysis of the clustering (higher score is better).
        silhouette_avg = silhouette_score(X_train, cluster_labels)
        print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
        generate_silhouette_plot(X_train, cluster_labels, k)

        # Analysis via Davies-Bouldin index (lower is better)
        score = davies_bouldin_score(X_train, cluster_labels)
        print("For n_clusters =", k, "The Davies Bouldin index is :", score)

        # From this basic analysis, it seems that k=3 or k=4 clusters is best.

# analyze_clustering()


def cluster_and_divide(num_clusters=3):
    clinical_tsv = "processed_data/clinical_train.tsv"
    clinical_df = pd.read_csv(clinical_tsv, sep="\t")

    gexp_top05_tsv = "processed_data/gene_expression_top05_train.tsv"
    gexp_df = pd.read_csv(gexp_top05_tsv, sep="\t")

    cox_data = cox.CoxRegressionDataset(gexp_df, clinical_df, standardize=False, test_size=0.0)
    X_train, y_train = cox_data.X, cox_data.y

    clusterer = KMeans(n_clusters=num_clusters, random_state=10)
    l_train = clusterer.fit_predict(X_train)

    for k in range(num_clusters):
        gexp_cluster_tsv = "clustering_analysis/gene_expression_top05_c%s_train.tsv" % k
        clinical_cluster_tsv = "clustering_analysis/clinical_c%s_train.tsv" % k

        c_indeces = np.where(l_train == k)


cluster_and_divide():
