
import pandas as pd
import numpy as np
from scipy.special import logsumexp

import sklearn as sk
from sklearn import decomposition, feature_selection, linear_model, preprocessing
from sksurv import linear_model as sk_lm

import matplotlib.pyplot as plt

import cox_regression_main as cox


################################################################
# Utility functions.
################################################################

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
    if 'Unnamed: 0' in feature_df.columns:
        print("Extra column found in feature_df")
        desired_columns.remove('Unnamed: 0')
    selected_df = feature_df.loc[:, desired_columns]
    return selected_df


def print_top_features_from_cox_coef(model_df, top_n=250, output_filename="output/cox_model_top_ranked_features.tsv"):
    visited_features = set([])
    ranked_features = []

    for row_i in range(model_df.shape[0]):
        # Process the row
        row_params, row_features = model_df.iloc[row_i, :], []
        for coef, feature_name in zip(row_params, row_params.index):
            if coef != 0:
                row_features.append(feature_name)

        rank = len(row_features)
        if rank > top_n:
            break

        for feature_name in row_features:
            if feature_name not in visited_features:
                visited_features.add(feature_name)
                ranked_features.append((feature_name, rank))

    # ranked_features.sort(reverse=False, key=lambda x: x[1])
    f = open(output_filename, "w")
    for feat, rank in ranked_features:
        f.write("%s\t%s\n" % (feat, rank))
    f.close()



def select_genes_from_paper(feature_df):
    """
    Return gene expression matrix with only the following genes:
    TAP1 - ENSG00000168394
    ZFHX4 - ENSG00000091656
    CXCL9 - ENSG00000138755
    FBN1 - ENSG00000166147
    PTGER3 - ENSG00000050628
    """
    gexp_df_top5_raw = feature_df[['case_id', 'ENSG00000168394.10', 'ENSG00000091656.14',
                          'ENSG00000138755.5', 'ENSG00000166147.12', 'ENSG00000050628.19']]
    # Normalize expression data
    scaler = preprocessing.StandardScaler()
    gexp_df_top5_normalized = pd.DataFrame(scaler.fit_transform(gexp_df_top5_raw.iloc[:, 1:6]))
    gexp_df_top5_normalized.insert(0, 'case_id', gexp_df_top5_raw['case_id'])
    #gexp_df_top5 = gexp_df_top5_id.merge(pd.DataFrame(gexp_df_top5_normalized))

    return gexp_df_top5_normalized






################################################################
# Test scripts.
################################################################

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




################################################################
# Staging scripts (routines to help stage data for downstream experiments)
################################################################

staged_data_dir = "processed_data/"

def variance_threshold(gexp_df, quantile=0.85, output_filename="gene_expression_top15_matrix.tsv"):
    """
    Select features in |gexp_df| via variance thresholding as specified by |quantile| and write to |output_filename|.
    """
    print("Num total features:", gexp_df.shape[1])
    gexp_selected = remove_low_variance_features(gexp_df, quantile=quantile, features_name="gene_expression")
    print("Num selected features:", gexp_selected.shape[1])
    gexp_selected.to_csv(output_filename, sep="\t", index=False)
    print("Variance-selected feature matrix written to " + output_filename + ".")
    return gexp_selected

gexp_tsv = "processed_data/gene_expression_matrix.tsv"
gexp_top15_tsv = staged_data_dir + "gene_expression_top15_matrix.tsv"
gexp_top05_tsv = staged_data_dir + "gene_expression_top05_matrix.tsv"
gexp_top03_tsv = staged_data_dir + "gene_expression_top03_matrix.tsv"
# print("\n-- Variance Thresholding Feature Selection --")
# gexp_df1 = pd.read_csv(gexp_tsv, sep="\t")
# gexp_df2 = variance_threshold(gexp_df1, quantile=0.95, output_filename=gexp_top05_tsv)
# gexp_df2 = variance_threshold(gexp_df1, quantile=0.85, output_filename=gexp_top15_tsv)
# gexp_df3 = variance_threshold(gexp_df1, quantile=0.97, output_filename=gexp_top03_tsv)



def coxnet_gexp_experiment(train_gexp_df, train_clinical_data, l1_ratio, output_filename="output/cox_model_gexp_exp.tsv"):
    """
    Trains a Cox Elastic Net model, whose learned parameters can then be used for ranking features.

    The resulting model file can be processed with the select_features_from_cox_coef() method above
    to get the top N number of features from the |gexp_df| dataset. The |l1_ratio| is for balancing
    the L1 and L2 regularizing terms (if l1_ratio=1.0, then we get ridge regression).
    """
    print("L1 ratio = %s, alpha_min_ratio = 0.01" % l1_ratio)
    train_dataset = cox.CoxRegressionDataset(train_gexp_df, train_clinical_data, standardize=True, test_size=0.0)

    coxnet_model = sk_lm.CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01)
    coxnet_model.fit(train_dataset.X, train_dataset.y)

    cox.save_cox_model(coxnet_model, train_dataset.feature_df, output_file=output_filename)
    # cox.basic_train_and_test(dataset, coxnet_model, model_file=output_filename)


exp_num = 5     # 3 or 5 is standard, use 8 next if another run is needed
# 3, 4 -- df, test_size=0.3 ?
# 5 -- train, test
# 6 -- train2, test2
# 7 -- train3, test3
cox_lasso_gexp = "output/cox_model_lasso_gexp_exp%s.tsv" % exp_num
cox_elast_gexp = "output/cox_model_elastic_gexp_exp%s.tsv" % exp_num
clinical_train_tsv = "processed_data/clinical_train.tsv"
gexp_top05_train_tsv = staged_data_dir + "gene_expression_top05_train.tsv"

# print("\n-- Feature Ranking based on Coefficients of ElasticNet-Regularized Cox Regression --")
# gexp_df = pd.read_csv(gexp_top05_train_tsv, sep="\t")
# clinical_df = pd.read_csv(clinical_train_tsv, sep="\t")
## coxnet_gexp_experiment(gexp_df, clinical_df, 1.0, output_filename=cox_lasso_gexp)     # Lasso
# coxnet_gexp_experiment(gexp_df, clinical_df, 0.9, output_filename=cox_elast_gexp)     # Elastic

# model_df = pd.read_csv(cox_elast_gexp, sep="\t", index_col=0)
# print_top_features_from_cox_coef(model_df, top_n=250, output_filename="output/cox_model_top_ranked_features.tsv")




# Copied from cox_nn.py for use below.
def compute_riskset_matrix(time):
    """
    Compute mask that represents each sample's risk set. Uses Breslow's method for tie-breakers.
    The risk set would be a boolean (n_samples, n_samples) matrix where the `i`-th row denotes the
    risk set of the `i`-th instance, i.e. the indices `j` for which the observer time `y_j >= y_i`.
    """
    # Sort in descending order.
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for ordered_i, time_i in enumerate(o):
        ti = time[time_i]
        k = ordered_i
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[time_i, o[:k]] = True
    return risk_set

def sure_independence_ranking(train_gexp_df, train_clinical_df, top_n=250, output_filename="output/sure_independence_ranking.tsv"):
    # For each feature:
    # Transform X into matrix with only the selected feature.
    # Train a CoxPH model on the modified X and y. This maximizes the partial likelihood wrt the specific feature.
    # Get the beta coefficient and use it to compute the likelihood value -> utility value.
    # Use utility values to rank the features.

    train_dataset = cox.CoxRegressionDataset(train_gexp_df, train_clinical_df, standardize=True, test_size=0.0)
    features = train_gexp_df.columns[1:]
    X, y = train_dataset.X, train_dataset.y

    event_field, time_field = y.dtype.names
    y_event, y_time = y[event_field], y[time_field]
    riskset = compute_riskset_matrix(y_time)

    num_samples, num_features = X.shape[0], X.shape[1]
    f_utils = []
    for j in range(num_features):
        if j % 250 == 0:
            print("Evaluated %s features" % j)

        # Train on this feature alone.
        X_j = X[:,j:j+1]
        cox_model = sk_lm.CoxPHSurvivalAnalysis()
        cox_model.fit(X_j, y)
        coef = cox_model.coef_[0]

        # Compute the marginal utility of the feature (the likelihood of this trained model).
        activations = X_j * coef
        activations_masked = np.multiply(activations.T, riskset)
        rr = logsumexp(activations_masked, axis=1, keepdims=True)
        assert rr.shape == activations.shape

        lklhds = np.multiply(np.array([y_event]).T, activations - rr)
        assert lklhds.shape == (num_samples, 1)
        total_likelihood = np.sum(lklhds)

        # Record the result.
        f_utils.append((j, total_likelihood))

    f_utils.sort(reverse=True, key=lambda x: x[1])

    f = open(output_filename, "w")
    num_features_recorded = 0
    for j, utility in f_utils:
        if num_features_recorded > top_n:
            break
        f.write("%s\t%s\n" % (features[j], utility))
        num_features_recorded += 1
    f.close()
        
    return f_utils


# clinical_train_tsv = "processed_data/clinical_train.tsv"
# gexp_top05_train_tsv = staged_data_dir + "gene_expression_top05_train.tsv"

# gexp_df = pd.read_csv(gexp_top05_train_tsv, sep="\t")
# clinical_df = pd.read_csv(clinical_train_tsv, sep="\t")
# sure_independence_ranking(gexp_df, clinical_df)



