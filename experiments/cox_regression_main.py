
import pandas as pd
import numpy as np

import sklearn as sk
from sklearn import model_selection
from sklearn import preprocessing

from sksurv import linear_model as sk_lm
from sksurv.util import Surv

import matplotlib.pyplot as plt



class CoxRegressionDataset(object):
    # Names for the columns corresponding to the labels used by this model.
    STATUS = "status"
    STATUS_TIME = "status_time"

    def __init__(self, feature_df, clinical_df, standardize=False, test_size=0.3):
        self.feature_df = feature_df
        self.clinical_df = clinical_df
        self.labels_df = self.get_labels(clinical_df)

        X, y = self.get_x_and_y(self.feature_df, self.labels_df)
        if standardize:
            X = self.standardize_data(X)
        if test_size > 0:
            self.X, self.X_test, self.y, self.y_test = self.split_dataset(X, y, test_size=test_size)
        else:
            self.X, self.y = X, y


    def standardize_data(self, X):
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        return X

    def get_labels(self, clinical_df):
        # Get only the relevant columns.
        labels_df = clinical_df[["case_id", "vital_status", "days_to_last_follow_up", "days_to_death"]]

        # Construct STATUS column for the Cox Regression model to use.
        # The label for each sample should be a structured array (status, event).
        def get_event(row):
            if np.isnan(row["days_to_last_follow_up"]):
                if np.isnan(row["days_to_death"]):
                    return 0.0
                return row["days_to_death"]
            elif np.isnan(row["days_to_death"]):
                return row["days_to_last_follow_up"]
            return max(row["days_to_last_follow_up"], row["days_to_death"])
        labels_df.loc[:, [self.STATUS_TIME]] = labels_df.apply(lambda row: get_event(row), axis=1)
        labels_df.loc[:, [self.STATUS]] = labels_df.apply(lambda row: (row["vital_status"] == "Dead"), axis=1)

        # Include only the processed columns.
        labels_df = labels_df[["case_id", self.STATUS, self.STATUS_TIME]]
        return labels_df

    def get_x_and_y(self, feature_df, labels_df):
        merged_df = labels_df.merge(feature_df, how="inner", on="case_id") #left_on="case_id", right_on="case_id")
        self.cases = merged_df["case_id"]

        # Gets the X and Y matrices for the model to use.
        y_dataframe = merged_df[[self.STATUS, self.STATUS_TIME]]
        y_vector = Surv.from_dataframe(self.STATUS, self.STATUS_TIME, y_dataframe)

        x_dataframe = merged_df.drop(["case_id", self.STATUS, self.STATUS_TIME], axis=1)
        x_matrix = np.asarray(x_dataframe)

        return x_matrix, y_vector

    def split_dataset(self, X, y, test_size):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=0)
        return X_train, X_test, y_train, y_test


def split_and_save_dataset(
    feature_df, clinical_df, test_size, train_X_filename, train_y_filename, test_X_filename, test_y_filename, random_state=None):
    merged_df = clinical_df.merge(feature_df, left_on="case_id", right_on="case_id")
    if random_state is None:
        shuffled_df = merged_df
    else:
        shuffled_df = merged_df.sample(frac=1, random_state=random_state)
    num_samples = shuffled_df.shape[0]
    
    y_df = shuffled_df[clinical_df.columns]
    drop_columns = clinical_df.columns.drop(["case_id"])
    x_df = shuffled_df.drop(drop_columns, axis=1)

    num_test_samples = int(np.rint(num_samples * test_size))
    x_test_df, y_test_df = x_df.iloc[:num_test_samples, :], y_df.iloc[:num_test_samples, :]
    x_train_df, y_train_df = x_df.iloc[num_test_samples:, :], y_df.iloc[num_test_samples:, :]

    print("Dataset split with test_size=%s" % test_size)

    x_test_df.to_csv(test_X_filename, sep="\t", index=False)
    print("Test data features written to %s." % test_X_filename)
    y_test_df.to_csv(test_y_filename, sep="\t", index=False)
    print("Test data labels written to %s." % test_y_filename)
    x_train_df.to_csv(train_X_filename, sep="\t", index=False)
    print("Training data features written to %s." % train_X_filename)
    y_train_df.to_csv(train_y_filename, sep="\t", index=False)
    print("Training data labels written to %s." % train_y_filename)

    return x_train_df, y_train_df, x_test_df, y_test_df


def save_cox_model(cox_model, feature_df, output_file="cox_exp1.tsv"):
    features = feature_df.drop(columns=["case_id"]).columns
    if len(cox_model.coef_.shape) == 1:
        coefficients = np.array([cox_model.coef_])
        model_df = pd.DataFrame(data=coefficients, columns=features)
    else:
        coefficients = cox_model.coef_.transpose()
        indeces = cox_model.alphas_
        model_df = pd.DataFrame(data=coefficients, index=indeces, columns=features)
    model_df.to_csv(output_file, sep="\t")
    print("Model parameters saved to " + output_file)


# def basic_train_and_test(cox_regression_dataset, cox_model, model_file="cox_model_exp1.tsv"):
#     data = cox_regression_dataset
#     cox_model = cox_model.fit(data.X, data.y)

#     train_score, test_score = cox_model.score(data.X, data.y), cox_model.score(data.X_test, data.y_test)
#     print("Concordance Index Censored for Training Dataset:", train_score)
#     print("Concordance Index Censored for Test Dataset:", test_score)

#     save_cox_model(cox_model, data.feature_df, output_file=model_file)

#     return train_score, test_score




##############################################################
# Run the below once to get a specific training/test split.
##############################################################

gexp_top05_tsv = "processed_data/gene_expression_top05_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed_adjusted.tsv"

# gexp_df = pd.read_csv(gexp_top05_tsv, sep="\t")
# clinical_df = pd.read_csv(clinical_tsv, sep="\t")

# X_train, y_train, X_test, y_test = split_and_save_dataset(
#     gexp_df, clinical_df, test_size=0.3,
#     train_X_filename="processed_data/gene_expression_top05_train.tsv",
#     train_y_filename="processed_data/clinical_train.tsv",
#     test_X_filename="processed_data/gene_expression_top05_test.tsv",
#     test_y_filename="processed_data/clinical_test.tsv")

## The below splits were done to do some debugging. Don't use these splits.
# X_train, y_train, X_test, y_test = split_and_save_dataset(
#     gexp_df, clinical_df, test_size=0.3,
#     train_X_filename="processed_data/gene_expression_top05_train2.tsv",
#     train_y_filename="processed_data/clinical_train2.tsv",
#     test_X_filename="processed_data/gene_expression_top05_test2.tsv",
#     test_y_filename="processed_data/clinical_test2.tsv",
#     random_state=10)
# X_train, y_train, X_test, y_test = split_and_save_dataset(
#     gexp_df, clinical_df, test_size=0.3,
#     train_X_filename="processed_data/gene_expression_top05_train3.tsv",
#     train_y_filename="processed_data/clinical_train3.tsv",
#     test_X_filename="processed_data/gene_expression_top05_test3.tsv",
#     test_y_filename="processed_data/clinical_test3.tsv")




##############################################################
# Run the below once to get a specific training/valid split.
##############################################################

train_gexp_top05_tsv = "processed_data/gene_expression_top05_train.tsv"
train_clinical_tsv = "processed_data/clinical_train.tsv"
# train_gexp_df = pd.read_csv(train_gexp_top05_tsv, sep="\t")
# train_clinical_df = pd.read_csv(train_clinical_tsv, sep="\t")

# X_strain, y_strain, X_valid, y_valid = split_and_save_dataset(
#     train_gexp_df, train_clinical_df, test_size=0.25,
#     train_X_filename="processed_data/gene_expression_top05_train_small.tsv",
#     train_y_filename="processed_data/clinical_train_small.tsv",
#     test_X_filename="processed_data/gene_expression_top05_valid.tsv",
#     test_y_filename="processed_data/clinical_valid.tsv")





"""
NOTE: The below scripting logic has been moved to feature_selection.py and cox_experiments.py for clarity.

Please go to the respective code locations instead for developing any additional experiments.
"""



# print("###### Mutations Data #######")

# print("-- 1. Reading Data --")

# mutation_tsv = "processed_data/mutations_matrix.tsv"
# clinical_tsv = "processed_data/clinical_processed.tsv"

# # mutation_df = pd.read_csv(mutation_tsv, sep="\t")
# # clinical_df = pd.read_csv(clinical_tsv, sep="\t")



# print("-- 2. Variance Thresholding Feature Selection --")

# # print("Num total features:", mutation_df.shape[1])
# # mutation_df2 = fs.remove_low_variance_features(mutation_df, quantile=0.85)
# # print("Num selected features:", mutation_df2.shape[1])



# # def initial_cox_experiment():
# #     dataset = CoxRegressionDataset(mutation_df2, clinical_df)
# #     cox_model = sk_lm.CoxPHSurvivalAnalysis(alpha=0.001, verbose=1)
# #     basic_train_and_test(dataset, cox_model, test_size=0.4, model_file="output/cox_model_exp1.tsv")
# # initial_cox_experiment()


# print("-- 3. Feature Selection based on Coefficients of Lasso-Regularized Cox Regression --")

# # This is all done with mutations data; we have yet to try gene expression data.
# # Reference: https://scikit-survival.readthedocs.io/en/latest/user_guide/coxnet.html#LASSO

# def coxnet_lasso_experiment():
#     dataset = CoxRegressionDataset(mutation_df2, clinical_df)
#     print("L1 ratio = 1.0, alpha_min_ratio = 0.01")
#     coxnet_model = sk_lm.CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01)
#     basic_train_and_test(dataset, coxnet_model, model_file="output/cox_model_lasso_exp2.tsv")
# # coxnet_lasso_experiment()

# def coxnet_lasso_feature_selection():
#     model_df = pd.read_csv("output/cox_model_lasso_exp2.tsv", sep="\t", index_col=0)
#     mutation_df3 = fs.select_features_from_cox_coef(model_df, mutation_df2, num_features=75)
#     mutation_df3.to_csv("processed_data/selected_lasso_83_mutations_matrix.tsv", sep="\t")
# # coxnet_lasso_feature_selection()


# print("-- 4. Feature Selection based on Coefficients of Elastic Net Cox Regression --")

# def coxnet_elastic_net_experiment():
#     dataset = CoxRegressionDataset(mutation_df2, clinical_df, standardize=True)
#     print("L1 ratio = 1.0, alpha_min_ratio = 0.01")
#     coxnet_model = sk_lm.CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01)
#     basic_train_and_test(dataset, coxnet_model, model_file="output/cox_model_elastic_exp1.tsv")
# # coxnet_elastic_net_experiment()

# def coxnet_elastic_net_feature_selection():
#     model_df = pd.read_csv("output/cox_model_elastic_exp1.tsv", sep="\t", index_col=0)
#     mutation_df3 = fs.select_features_from_cox_coef(model_df, mutation_df2, num_features=75)
#     mutation_df3.to_csv("processed_data/selected_elastic_78_mutations_matrix.tsv", sep="\t")
# # coxnet_elastic_net_feature_selection()


# print("-- 5. Run Cox Regression Cross-Validated Experiment with Selected Features --")

# def cox_experiment_with_selected_features():
#     mutation_df3 = pd.read_csv("processed_data/selected_lasso_83_mutations_matrix.tsv", sep="\t")
#     # mutation_df3 = pd.read_csv("processed_data/selected_elastic_78_mutations_matrix.tsv", sep="\t")
#     print("Num selected features:", mutation_df3.shape[1])
#     dataset = CoxRegressionDataset(mutation_df3, clinical_df)

#     alphas = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 3.0, 5.0, 10.0]
#     # alphas = [0.005, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
#     models = [sk_lm.CoxPHSurvivalAnalysis(alpha=a) for a in alphas]
#     scores = [np.mean(model_selection.cross_val_score(model, dataset.X, dataset.y)) for model in models]
#     print(scores)
#     max_score, argmax_score = 0, None
#     for i, score in enumerate(scores):
#         if score > max_score:
#             max_score, argmax_score = score, i
#     model = models[argmax_score]
#     model.fit(dataset.X, dataset.y)
#     print("Using alpha=%s:\nAverage Cross-Validation Score=\t%s\nTest Score=\t\t\t%s" % (
#         alphas[argmax_score], score, model.score(dataset.X_test, dataset.y_test)) )

# # cox_experiment_with_selected_features()



# print("\n###### Gene Expression Data #######")

# print("\n-- 1. Variance Thresholding Feature Selection --")

# clinical_tsv = "processed_data/clinical_processed.tsv"
# clinical_df = pd.read_csv(clinical_tsv, sep="\t")

# def variance_threshold(gexp_df, quantile=0.85, output_filename="gene_expression_top3_matrix.tsv"):
#     print("Num total features:", gexp_df.shape[1])
#     gexp_df2 = fs.remove_low_variance_features(gexp_df, quantile=quantile, features_name="gene_expression")
#     print("Num selected features:", gexp_df2.shape[1])
#     gexp_df2.to_csv(output_filename, sep="\t", index=False)
#     print("Variance-selected feature matrix written to " + output_filename + ".")
#     return gexp_df2

# # gexp_tsv = "processed_data/gene_expression_matrix.tsv"
# # gexp_df1 = pd.read_csv(gexp_tsv, sep="\t")
# # gexp_top05_tsv = "processed_data/gene_expression_top05_matrix.tsv"
# # gexp_df2 = variance_threshold(gexp_df1, quantile=0.95, output_filename=gexp_top05_tsv)
# # gexp_top15_tsv = "processed_data/gene_expression_top15_matrix.tsv"
# # gexp_df2 = variance_threshold(gexp_df1, quantile=0.85, output_filename=gexp_top15_tsv)
# # gexp_top03_tsv = "processed_data/gene_expression_top03_matrix.tsv"
# # gexp_df3 = variance_threshold(gexp_df1, quantile=0.97, output_filename=gexp_top03_tsv)



# print("\n-- 2. Feature Ranking based on Coefficients of Lasso-Regularized Cox Regression --")

# def coxnet_gexp_experiment(gexp_df, l1_ratio, output_filename="output/cox_model_gexp_exp.tsv"):
#     print("Using Ridge Regression for Cox Regression (L1 Regularization) for feature selection.")
#     print("  L1 ratio = %s, alpha_min_ratio = 0.01" % l1_ratio)
#     dataset = CoxRegressionDataset(gexp_df, clinical_df, standardize=True)
#     coxnet_model = sk_lm.CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01)
#     basic_train_and_test(dataset, coxnet_model, model_file=output_filename)

# exp_num = 3
# cox_lasso_gexp = "output/cox_model_lasso_gexp_exp%s.tsv" % exp_num
# cox_elast_gexp = "output/cox_model_elastic_gexp_exp%s.tsv" % exp_num
# # gexp_df2 = pd.read_csv(gexp_top05_tsv, sep="\t")
# # coxnet_gexp_experiment(gexp_df2, 1.0, output_filename=cox_lasso_gexp)
# # coxnet_gexp_experiment(gexp_df2, 0.9, output_filename=cox_elast_gexp)


# print("\n-- 3. Select a feature set from list provided by previous Cox Regression feature selection process. --")

# def cox_experiment_with_selected_gexp_features(gexp_df, model_df, num_features=77, alpha_score_plot=None):
#     print("Selecting a set of %s features from the coxnet results." % num_features)
#     selected_gexp_df = fs.select_features_from_cox_coef(model_df, gexp_df, num_features=num_features)
#     dataset = CoxRegressionDataset(selected_gexp_df, clinical_df, standardize=True)

#     # Instead of picking arbitrary alphas, select range of alphas based on an initial coxnet run
#     small_model = sk_lm.CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, l1_ratio=0.95, max_iter=10000)
#     small_model.fit(dataset.X, dataset.y)
#     alphas = list(reversed(list(small_model.alphas_)[0::5]))  # select every 5th alpha in the range.
#     print("Selected alphas for this run:", alphas)

#     # Train models on the different alpha values. Evaluate via cross-validation.
#     models = [sk_lm.CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=0.95) for a in alphas]
#     results = [model_selection.cross_validate(model, dataset.X, dataset.y, return_train_score=True) for model in models]

#     # Get scores and standard deviations.
#     valid_scores = [np.mean(result["test_score"]) for result in results]
#     valid_stds = [np.std(result["test_score"]) for result in results]
#     train_scores = [np.mean(result["train_score"]) for result in results]
#     train_stds = [np.std(result["train_score"]) for result in results]
#     print("Mean Cross-Validation Scores:", valid_scores)
#     print("Std Dev Cross-Validation Scores:", valid_stds)

#     # Record metrics for each model.
#     max_score, argmax_score = 0, None
#     for i, score in enumerate(valid_scores):
#         if score > max_score:
#             max_score, argmax_score = score, i

#     # Test score.
#     model = models[argmax_score]
#     model.fit(dataset.X, dataset.y)
#     test_score = model.score(dataset.X_test, dataset.y_test)
#     # save_cox_model(model, selected_gexp_df, output_file="output/selected_coxnet_model3.tsv")

#     # Plot alpha vs validation score
#     if alpha_score_plot is not None:
#         plt.cla()
#         plt.plot(alphas, valid_scores, c="blue")
#         plt.plot(alphas, train_scores, c="orange")
#         plt.savefig(alpha_score_plot)

#     print("Using alpha=%s:\nAverage Cross-Validation Score=\t%s\nTest Score=\t\t\t%s\n" % (
#         alphas[argmax_score], max_score, test_score) )
#     return max_score, train_scores[argmax_score], test_score, alphas[argmax_score]




# Script for the gene expression experiment moved to cox_experiment.py

