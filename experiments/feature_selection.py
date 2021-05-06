
import pandas as pd

import numpy as np
import sklearn as sk
from sklearn import feature_selection

import matplotlib.pyplot as plt


def remove_low_variance_features(feature_df, quantile=0.85, features_name="feature"):
    variances = np.var(np.asarray(feature_df)[:,1:], axis=0)
    quant = np.quantile(variances, quantile)

    plt.hist(variances, bins=50)
    plt.yscale("log")
    plt.title("Histogram of " + features_name + " variances\n(red line is 85th percentile)")
    plt.xlabel("Variance of Feature")
    plt.ylabel("Num features with specified variance (log)")
    plt.vlines(quant, colors="red", ymin=0, ymax=plt.gca().get_ylim()[1])
    plt.savefig(features_name + "_variance.png")

    sel = feature_selection.VarianceThreshold(threshold=quant)
    sel.fit(feature_df.drop(columns=["case_id"]))
    retained_indeces = sel.get_support(indices=True)
    new_feature_df = feature_df.iloc[:,retained_indeces]

    return new_feature_df



mutation_tsv = "processed_data/mutations_matrix.tsv"
gexp_tsv = "processed_data/gene_expression_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed.tsv"

raw_mut_df = pd.read_csv(mutation_tsv, sep="\t")
mut_df = remove_low_variance_features(raw_mut_df, features_name="mutation_feature")

