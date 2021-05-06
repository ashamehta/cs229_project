
import pandas as pd
import numpy as np

import sklearn as sk
from sklearn import model_selection
from sklearn import preprocessing

from sksurv import linear_model as sk_lm
from sksurv.util import Surv

import matplotlib.pyplot as plt

import feature_selection as fs


class CoxRegressionDataset(object):
    # Names for the columns corresponding to the labels used by this model.
    STATUS = "status"
    STATUS_TIME = "status_time"

    def __init__(self, feature_df, clinical_df, test_size=0.4):
        self.feature_df = feature_df
        self.labels_df = self.get_labels(clinical_df)

        X, y = self.get_x_and_y(self.feature_df, self.labels_df)
        self.X, self.X_test, self.y, self.y_test = self.split_dataset(X, y, test_size=test_size)

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
        labels_df[self.STATUS_TIME] = labels_df.apply(lambda row: get_event(row), axis=1)
        labels_df[self.STATUS] = labels_df.apply(lambda row: (row["vital_status"] == "Dead"), axis=1)

        # Include only the processed columns.
        labels_df = labels_df[["case_id", self.STATUS, self.STATUS_TIME]]
        return labels_df

    def get_x_and_y(self, feature_df, labels_df):
        merged_df = labels_df.merge(feature_df, left_on="case_id", right_on="case_id")

        # Gets the X and Y matrices for the model to use.
        y_dataframe = merged_df[[self.STATUS, self.STATUS_TIME]]
        y_vector = Surv.from_dataframe(self.STATUS, self.STATUS_TIME, y_dataframe)

        x_dataframe = merged_df.drop(["case_id", self.STATUS, self.STATUS_TIME], axis=1)
        x_matrix = np.asarray(x_dataframe)

        return x_matrix, y_vector

    def split_dataset(self, X, y, test_size):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=0)
        return X_train, X_test, y_train, y_test


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


def basic_train_and_test(cox_regression_dataset, cox_model, model_file="cox_model_exp1.tsv"):
    data = cox_regression_dataset
    cox_model = cox_model.fit(data.X, data.y)

    train_score, test_score = cox_model.score(data.X, data.y), cox_model.score(data.X_test, data.y_test)
    print("Concordance Index Censored for Training Dataset:", train_score)
    print("Concordance Index Censored for Test Dataset:", test_score)

    save_cox_model(cox_model, data.feature_df, output_file=model_file)

    return train_score, test_score


def cross_validation_tune(cox_regression_dataset, cox_models):
    data = cox_regression_dataset
    scores = [model_selection.cross_val_score(model, data.X, data.y) for model in cox_models]
    return scores





print("-- 1. Reading Data --")

mutation_tsv = "processed_data/mutations_matrix.tsv"
gexp_tsv = "processed_data/gene_expression_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed.tsv"

mutation_df = pd.read_csv(mutation_tsv, sep="\t")
gexp_df = None
clinical_df = pd.read_csv(clinical_tsv, sep="\t")



print("-- 2. Variance Thresholding Feature Selection --")

print("Num total features:", mutation_df.shape[1])
mutation_df2 = fs.remove_low_variance_features(mutation_df, quantile=0.85)
print("Num selected features:", mutation_df2.shape[1])



print("-- 3. Feature Selection based on Coefficients of Lasso-Regularized Cox Regression --")

# This is all done with mutations data; we have yet to try gene expression data.
# Reference: https://scikit-survival.readthedocs.io/en/latest/user_guide/coxnet.html#LASSO

def initial_cox_experiment():
    dataset = CoxRegressionDataset(mutation_df2, clinical_df)
    cox_model = sk_lm.CoxPHSurvivalAnalysis(alpha=0.001, verbose=1)
    basic_train_and_test(dataset, cox_model, test_size=0.4, model_file="output/cox_model_exp1.tsv")
# initial_cox_experiment()

def coxnet_lasso_experiment():
    dataset = CoxRegressionDataset(mutation_df2, clinical_df)
    print("L1 ratio = 1.0, alpha_min_ratio = 0.01")
    coxnet_model = sk_lm.CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01)
    basic_train_and_test(dataset, coxnet_model, model_file="output/cox_model_lasso_exp2.tsv")
# coxnet_lasso_experiment()

def coxnet_lasso_feature_selection():
    model_df = pd.read_csv("cox_model_lasso_exp2.tsv", sep="\t", index_col=0)
    mutation_df3 = fs.select_features_from_cox_coef(model_df, mutation_df2, num_features=75)
    mutation_df3.to_csv("processed_data/selected_mutations_matrix.tsv", sep="\t")
# coxnet_lasso_feature_selection()

def cox_experiment_with_selected_features():
    mutation_df3 = pd.read_csv("processed_data/selected_mutations_matrix.tsv", sep="\t")
    print("Num selected features:", mutation_df3.shape[1])
    dataset = CoxRegressionDataset(mutation_df3, clinical_df)

    alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03]
    models = [sk_lm.CoxPHSurvivalAnalysis(alpha=a) for a in alphas]
    scores = [np.mean(model_selection.cross_val_score(model, dataset.X, dataset.y)) for model in models]
    print(scores)
    for alpha, model, score in zip(alphas, models, scores):
        if score == max(scores):
            model.fit(dataset.X, dataset.y)
            print("Using alpha=%s:\nAverage Cross-Validation Score=%s\nAverage Cross-Validation Score=%s" % (
                alpha, score, model.score(dataset.X_test, dataset.y_test)) )
            break

cox_experiment_with_selected_features()


# TODO: Repeat above with gene expression data.






