
import pandas as pd
import numpy as np

import sklearn as sk
from sklearn import model_selection
from sklearn import preprocessing

from sksurv import linear_model as sk_lm
from sksurv.util import Surv

import matplotlib.pyplot as plt

import feature_selection as fs


class CoxRegressionExp(object):
    # Names for the columns corresponding to the labels used by this model.
    STATUS = "status"
    STATUS_TIME = "status_time"

    def __init__(self, feature_df, clinical_df,
                 model=sk_lm.CoxPHSurvivalAnalysis(alpha=0.001, verbose=1)):
        self.feature_df = feature_df
        self.labels_df = self.get_labels(clinical_df)
        self.model = model

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

    def get_x_and_y(self, merged_dataframe):
        # Gets the X and Y matrices for the model to use.
        y_dataframe = merged_dataframe[[self.STATUS, self.STATUS_TIME]]
        y_vector = Surv.from_dataframe(self.STATUS, self.STATUS_TIME, y_dataframe)

        x_dataframe = merged_dataframe.drop(["case_id", self.STATUS, self.STATUS_TIME], axis=1)
        x_matrix = np.asarray(x_dataframe)

        return x_matrix, y_vector

    def save_model(self, output_file="cox_model_exp1.tsv"):
        features = self.feature_df.drop(columns=["case_id"]).columns
        coefficients = np.array([self.model.coef_])
        model_df = pd.DataFrame(data=coefficients, columns=features)
        model_df.to_csv(output_file, sep="\t")
        print("Model parameters saved to " + output_file)

    def run_experiment(self, model_file="cox_model_exp1.tsv"):
        # Order of rows matters from this point on (ensures data for cases are aligned).
        merged_df = self.labels_df.merge(self.feature_df, left_on="case_id", right_on="case_id")
        X, y = self.get_x_and_y(merged_df)
        # print(y)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4, random_state=0)
        self.model = self.model.fit(X_train, y_train)

        train_score, test_score = self.model.score(X_train, y_train), self.model.score(X_test, y_test)
        print("Concordance Index Censored for Training Dataset:", train_score)
        print("Concordance Index Censored for Test Dataset:", test_score)

        self.save_model()

        return test_score


mutation_tsv = "processed_data/mutations_matrix.tsv"
gexp_tsv = "processed_data/gene_expression_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed.tsv"

def read_data(mutation_tsv, gexp_tsv, clinical_tsv):
    # Read input data.
    mut_df = pd.read_csv(mutation_tsv, sep="\t")
    # gexp_df = pd.read_csv(gexp_tsv, sep="\t")
    gexp_df = None
    clin_df = pd.read_csv(clinical_tsv, sep="\t")
    return mut_df, gexp_df, clin_df

print("-- Reading Data --")
mutation_df, gexp_df, clinical_df = read_data(mutation_tsv, gexp_tsv, clinical_tsv)

print("-- Some Basic Feature Selection --")
print("Num total features:", mutation_df.shape[1])
mutation_sel_df = fs.remove_low_variance_features(mutation_df, quantile=0.85)
print("Num selected features:", mutation_sel_df.shape[1])

print("-- Running Experiment --")
def basic_cox_experiment():
    cox_experiment = CoxRegressionExp(mutation_sel_df, clinical_df)
    mut_test_score = cox_experiment.run_experiment(model_file="cox_model_exp2.tsv")
# basic_cox_experiment()

def coxnet_lasso_experiment():
    print("L1 ratio = 1.0, alpha_min_ratio = 0.01")
    coxnet_model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01)
    cox_experiment = CoxRegressionExp(mutation_sel_df, clinical_df, model=coxnet_model)
    mut_test_score = cox_experiment.run_experiment(model_file="cox_model_lasso_exp2.tsv")


