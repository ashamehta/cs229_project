
import pandas as pd

import numpy as np
import sklearn as sk
from sklearn import decomposition, feature_selection, linear_model

import matplotlib.pyplot as plt

from genetic_selection import GeneticSelectionCV


def remove_low_variance_features(feature_df, threshold=None, quantile=0.97, features_name=None):
    """Applies variance thresholding to feature_df, eliminating features with variance under the |quantile|."""
    variances = np.var(np.asarray(feature_df)[:,1:], axis=0)
    if threshold is None:
        threshold = np.quantile(variances, quantile)
    print("Variance threshold:", threshold)

    if features_name is not None:
        plt.hist(variances, bins=100)
        plt.yscale("log")
        plt.title("Histogram of " + features_name + " variances\n(red line is selected threshold)")
        plt.xlabel("Variance of Feature")
        plt.ylabel("Num features with specified variance (log)")
        plt.vlines(threshold, colors="red", ymin=0, ymax=plt.gca().get_ylim()[1])
        plt.savefig(features_name + "_variance.png")

    sel = feature_selection.VarianceThreshold(threshold=threshold)

    feature_only_df = feature_df.drop(columns=["case_id"])
    sel.fit(feature_only_df)
    retained_indeces = sel.get_support(indices=True)
    retained_features = feature_only_df.columns[retained_indeces]

    new_feature_df = feature_df.loc[:, ["case_id"] + retained_features.tolist()]

    return new_feature_df


# Copied from https://scikit-survival.readthedocs.io/en/latest/user_guide/coxnet.html
def plot_coefficients(coefs, n_highlight):
    """Plots each feature's corresponding coefficient in the Cox model as alpha changes."""
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)
    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min, coef, name + "   ",
            horizontalalignment="right",
            verticalalignment="center"
        )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")


def select_features_from_cox_coef(coef_df, feature_df, num_features):
    num_nonzero_features = np.sum(np.sign(np.abs(np.asarray(coef_df))), axis=1)
    for num, alpha in zip(num_nonzero_features, coef_df.index):
        if num >= num_features:
            print("Selected " + str(num) + " number of features at alpha=" + str(alpha))
            break
    selected_params, selected_features = coef_df.loc[alpha,:], []
    for coef, feature in zip(selected_params, selected_params.index):
        if coef != 0:
            selected_features.append(feature)

    desired_columns = ["case_id"] + selected_features
    desired_columns.remove('Unnamed: 0')
    selected_df = feature_df.loc[:, desired_columns]
    return selected_df

def select_genes_from_paper(feature_df):
    """
    Return gene expression matrix with only the following genes:
    TAP1 - ENSG00000168394
    ZFHX4 - ENSG00000091656
    CXCL9 - ENSG00000138755
    FBN1 - ENSG00000166147
    PTGER3 - ENSG00000050628
    """
    gexp_df = feature_df[['case_id', 'ENSG00000168394.10', 'ENSG00000091656.14',
                          'ENSG00000138755.5', 'ENSG00000166147.12', 'ENSG00000050628.19']]
    return gexp_df

# # Test out variance thresholding.
mutation_tsv = "processed_data/mutations_matrix.tsv"
gexp_tsv = "processed_data/gene_expression_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed.tsv"

# raw_mut_df = pd.read_csv(mutation_tsv, sep="\t")
# mut_df = remove_low_variance_features(raw_mut_df, features_name="mutation_feature")

# raw_gexp_df = pd.read_csv(gexp_tsv, sep="\t")
# gexp_df = remove_low_variance_features(raw_gexp_df, quantile=0.95, features_name="gene_expression_feature")


# # Select features based on feature coefficients from Cox Regression with lasso regularization.
# mutation_tsv = "processed_data/mutations_matrix.tsv"
# model_file = "cox_model_lasso_exp2.tsv"
#
# mut_df = pd.read_csv(mutation_tsv, sep="\t")
# model_df = pd.read_csv(model_file, sep="\t", index_col=0)
#
# sel_mut_df = select_features_from_cox_coef(model_df, mut_df)
# sel_mut_df.to_csv("processed_data/selected_mutations_matrix.tsv", sep="\t")





