
import pandas as pd
import numpy as np

import sklearn as sk
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis

import matplotlib.pyplot as plt


class CoxRegressionExp(object):
    # Names for the columns corresponding to the labels used by this model.
    LABEL_NAME = "label"

    def __init__(self, feature_df, clinical_df):
        self.feature_df = feature_df
        self.labels_df = self.get_labels(clinical_df)
        self.model = CoxPHSurvivalAnalysis()

    def get_labels(self, clinical_df):
        # Get only the relevant columns.
        labels_df = clinical_df[["case_id", "vital_status", "days_to_last_follow_up", "days_to_death"]]

        # Construct label column for the Cox Regression model to use.
        # The label for each sample should be a structured array (status, event).
        def get_event(row):
            # Each row would be labeled with (status, status event)
            status = (row["vital_status"] == "Dead")
            if np.isnan(row["days_to_last_follow_up"]):
                if np.isnan(row["days_to_death"]):
                    return (status, 0.0)
                return (status, row["days_to_death"])
            elif np.isnan(row["days_to_death"]):
                return (status, row["days_to_last_follow_up"])
            return (status, max(row["days_to_last_follow_up"], row["days_to_death"]))
        labels_df[self.LABEL_NAME] = labels_df.apply(lambda row: get_event(row), axis=1)

        # Include only the processed columns.
        labels_df = labels_df[["case_id", self.LABEL_NAME]]
        return labels_df

    def get_x_and_y(self, merged_dataframe):
        # Gets the X and Y matrices for the model to use.
        y_dataframe = merged_dataframe[[self.LABEL_NAME]]
        print(y_dataframe)
        x_dataframe = merged_dataframe.drop(["case_id", self.LABEL_NAME], axis=1)
        return np.asarray(x_dataframe), np.asarray(y_dataframe)

    def run_experiment(self):
        # Order of rows matters from this point on (ensures data for cases are aligned).
        merged_df = self.labels_df.merge(self.feature_df, left_on="case_id", right_on="case_id")
        X, y = self.get_x_and_y(merged_df)
        # print(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
        model = self.model.fit(X_train, y_train)

        train_score, test_score = model.score(X_train, y_train), model.score(X_test, y_test)
        print("Concordance Index Censored for Training Dataset:", train_score)
        print("Concordance Index Censored for Test Dataset:", test_score)

        return test_score


mutation_tsv = "processed_data/mutations_matrix.tsv"
gexp_tsv = "processed_data/gene_expression_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed.tsv"

def read_data(mutation_tsv, gexp_tsv, clinical_tsv):
    # Read
    mut_df = pd.read_csv(mutation_tsv, sep="\t")
    # gexp_df = pd.read_csv(gexp_tsv, sep="\t")
    gexp_df = None
    clin_df = pd.read_csv(clinical_tsv, sep="\t")
    return mut_df, gexp_df, clin_df

print("-- Reading Data --")
mutation_df, gexp_df, clinical_df = read_data(mutation_tsv, gexp_tsv, clinical_tsv)

print("-- Running Experiment --")
cox_experiment = CoxRegressionExp(mutation_df, clinical_df)
mut_test_score = cox_experiment.run_experiment()

