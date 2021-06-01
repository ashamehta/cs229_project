
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

import cox_regression_main as cox
import feature_selection as fs
import clustering as cl

import cox_experiments as cox_exp


def cluster_data(train=True, clusterer=None, num_clusters=3):
    suffix = "train" if train else "test"
    gexp_top05_tsv = "processed_data/gene_expression_top05_%s.tsv" % suffix
    gexp_df = pd.read_csv(gexp_top05_tsv, sep="\t")
    gexp_df = gexp_df.drop_duplicates(subset=["case_id"])

    clinical_tsv = "processed_data/clinical_%s.tsv" % suffix
    clinical_df = pd.read_csv(clinical_tsv, sep="\t")
    clinical_df = clinical_df.drop_duplicates(subset=["case_id"])

    cox_data = cox.CoxRegressionDataset(gexp_df, clinical_df, standardize=False, test_size=0.0)
    X, cases = cox_data.X, np.array(list(cox_data.cases))

    if train:
        clusterer = KMeans(n_clusters=num_clusters, random_state=10)
        labels = clusterer.fit_predict(X)
    else:
        assert clusterer is not None
        labels = clusterer.predict(X)

    clustered_data_splits = []
    for k in range(num_clusters):
        gexp_cluster_tsv = "clustering_analysis/gene_expression_top05_c%s_%s2.tsv" % (k, suffix)
        clinical_cluster_tsv = "clustering_analysis/clinical_c%s_%s2.tsv" % (k, suffix)

        c_indeces = np.where(labels == k)[0]
        c_cases = cases[c_indeces]
        gexp_cluster_df = gexp_df.loc[gexp_df['case_id'].isin(c_cases), :]
        clinical_cluster_df = clinical_df.loc[clinical_df['case_id'].isin(c_cases), :]

        gexp_cluster_df.to_csv(gexp_cluster_tsv, sep="\t", index=False)
        clinical_cluster_df.to_csv(clinical_cluster_tsv, sep="\t", index=False)

        clustered_data_splits.append((gexp_cluster_df, clinical_cluster_df))
        print("Cluster %s has %s number of %s samples." % (k, gexp_cluster_df.shape[0], suffix))

    return clustered_data_splits, clusterer


def learn_coxnet_on_clustered_data(num_clusters=3):
    for k in range(num_clusters):
        gexp_cluster_tsv = "clustering_analysis/gene_expression_top05_c%s_train2.tsv" % k 
        clinical_cluster_tsv = "clustering_analysis/clinical_c%s_train2.tsv" % k 

        gexp_cluster_df = pd.read_csv(gexp_cluster_tsv, sep="\t")
        clinical_cluster_df = pd.read_csv(clinical_cluster_tsv, sep="\t")
        output_file = "clustering_analysis/cox_model_elastic_c%s_exp2.tsv" % k
        fs.coxnet_gexp_experiment(gexp_cluster_df, clinical_cluster_df, 0.9, output_filename=output_file)


def si_rank_on_clustered_data(num_clusters=3):
    for k in range(num_clusters):
        gexp_cluster_tsv = "clustering_analysis/gene_expression_top05_c%s_train2.tsv" % k 
        clinical_cluster_tsv = "clustering_analysis/clinical_c%s_train2.tsv" % k 

        gexp_cluster_df = pd.read_csv(gexp_cluster_tsv, sep="\t")
        clinical_cluster_df = pd.read_csv(clinical_cluster_tsv, sep="\t")
        output_file = "clustering_analysis/si_ranking_c%s_exp2.tsv" % k
        fs.sure_independence_ranking(gexp_cluster_df, clinical_cluster_df, top_n=250, output_filename=output_file)


# RUN 1: Simple cluster on training data, observe any differences in coxnet result.

# data_splits, clusterer = cluster_train_data(train=True, num_clusters=3)
# test_clusters, clusterer = cluster_train_data(train=False, clusterer=clusterer, num_clusters=3)
# learn_coxnet_on_clustered_data()

# gexp_df = pd.read_csv("processed_data/gene_expression_top05_train.tsv", sep="\t")
# cox_net_c0_model_df = pd.read_csv("clustering_analysis/cox_model_elastic_c0_exp.tsv", sep="\t", index_col=0)
# cox_net_c1_model_df = pd.read_csv("clustering_analysis/cox_model_elastic_c1_exp.tsv", sep="\t", index_col=0)
# cox_net_c2_model_df = pd.read_csv("clustering_analysis/cox_model_elastic_c2_exp.tsv", sep="\t", index_col=0)
# selected_c0_df = fs.select_features_from_cox_coef(cox_net_c0_model_df, gexp_df, num_features=75)
# selected_c1_df = fs.select_features_from_cox_coef(cox_net_c1_model_df, gexp_df, num_features=75)
# selected_c2_df = fs.select_features_from_cox_coef(cox_net_c2_model_df, gexp_df, num_features=75)


# selected_c0_df = selected_c0_df.drop(["case_id"], axis=1)
# selected_c1_df = selected_c1_df.drop(["case_id"], axis=1)
# selected_c2_df = selected_c2_df.drop(["case_id"], axis=1)
# # print(selected_net_df)
# c0_features = set(selected_c0_df.columns)
# c1_features = set(selected_c1_df.columns)
# c2_features = set(selected_c2_df.columns)

# common_features = c0_features.intersection(c2_features)
# print(common_features)


# RUN 2:

## Create the clustered data splits.
# data_splits, clusterer = cluster_data(train=True, num_clusters=3)
# test_clusters, clusterer = cluster_data(train=False, clusterer=clusterer, num_clusters=3)

## Use CoxNet to rank features for each cluster (except cluster 0, which has not enough data...) 
# learn_coxnet_on_clustered_data()

## Use SI Ranking to rank features for each cluster
si_rank_on_clustered_data(num_clusters=3)

## Analyze the different models for each feature set for each cluster (except cluster 0...)
# for k in [1, 2]:
#     train_gexp_tsv = "clustering_analysis/gene_expression_top05_c%s_train2.tsv" % k
#     train_clinical_tsv = "clustering_analysis/clinical_c%s_train2.tsv" % k
#     test_gexp_tsv = "clustering_analysis/gene_expression_top05_c%s_test2.tsv" % k
#     test_clinical_tsv = "clustering_analysis/clinical_c%s_test2.tsv" % k
#     model_tsv = "clustering_analysis/cox_model_elastic_c%s_exp2.tsv" % k

#     train_gexp_df = pd.read_csv(train_gexp_tsv, sep="\t")
#     train_clinical_df = pd.read_csv(train_clinical_tsv, sep="\t")
#     test_gexp_df = pd.read_csv(test_gexp_tsv, sep="\t")
#     test_clinical_df = pd.read_csv(test_clinical_tsv, sep="\t")
#     model_df = pd.read_csv(model_tsv, sep="\t", index_col=0)

#     print("Cluster " + str(k))
#     print(train_gexp_df.shape)
#     print(train_clinical_df.shape)
#     print(test_gexp_df.shape)
#     print(test_clinical_df.shape)
#     assert False
#     # print(model_df)
#     # num_nonzero_features = np.sum(np.sign(np.abs(np.asarray(model_df))), axis=1)
#     # print(num_nonzero_features)
#     # assert False

#     output_file = "clustering_analysis/model_selection_c%s_run2_log.txt" % k
#     cox_exp.iterative_model_selection(train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df, model_df, output_file)



# gexp_df = pd.read_csv("processed_data/gene_expression_top05_train.tsv", sep="\t")
# cox_net_c0_model_df = pd.read_csv("clustering_analysis/cox_model_elastic_c0_exp2.tsv", sep="\t", index_col=0)
# cox_net_c1_model_df = pd.read_csv("clustering_analysis/cox_model_elastic_c1_exp2.tsv", sep="\t", index_col=0)
# cox_net_c2_model_df = pd.read_csv("clustering_analysis/cox_model_elastic_c2_exp2.tsv", sep="\t", index_col=0)

# selected_c0_df = fs.select_features_from_cox_coef(cox_net_c0_model_df, gexp_df, num_features=75)
# selected_c1_df = fs.select_features_from_cox_coef(cox_net_c1_model_df, gexp_df, num_features=75)
# selected_c2_df = fs.select_features_from_cox_coef(cox_net_c2_model_df, gexp_df, num_features=75)

# selected_c0_df = selected_c0_df.drop(["case_id"], axis=1)
# selected_c1_df = selected_c1_df.drop(["case_id"], axis=1)
# selected_c2_df = selected_c2_df.drop(["case_id"], axis=1)
# # print(selected_net_df)
# c0_features = set(selected_c0_df.columns)
# c1_features = set(selected_c1_df.columns)
# c2_features = set(selected_c2_df.columns)

# common_features = c1_features.intersection(c2_features)
# print(common_features)

