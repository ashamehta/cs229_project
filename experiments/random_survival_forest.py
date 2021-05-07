import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sksurv.util import Surv

from sksurv.ensemble import RandomSurvivalForest


mutation_tsv = "processed_data/mutations_matrix.tsv"
gexp_tsv = "processed_data/gene_expression_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed.tsv"

RANDOM_STATE = 20

def read_data(mutation_tsv, gexp_tsv, clinical_tsv):
    # Read
    mut_df = pd.read_csv(mutation_tsv, sep="\t")
    gexp_df = pd.read_csv(gexp_tsv, sep="\t")
    # gexp_df = None
    clin_df = pd.read_csv(clinical_tsv, sep="\t")
    return mut_df, gexp_df, clin_df

def get_labels_and_merge(clinical_df, feature_df):
    # Get only the relevant columns.
    labels_df = clinical_df[["case_id", "vital_status", "days_to_last_follow_up", "days_to_death"]]

def get_labels(clinical_df):
    # Get only the relevant columns.
    labels_df = clinical_df[["case_id", "vital_status", "days_to_last_follow_up", "days_to_death"]]

    # Construct STATUS column for the RSF model to use.
    # The label for each sample should be a structured array (status, event).
    def get_event(row):
        if np.isnan(row["days_to_last_follow_up"]):
            if np.isnan(row["days_to_death"]):
                return 0.0
            return row["days_to_death"]
        elif np.isnan(row["days_to_death"]):
            return row["days_to_last_follow_up"]
        return max(row["days_to_last_follow_up"], row["days_to_death"])
    labels_df["status_time"] = labels_df.apply(lambda row: get_event(row), axis=1)
    labels_df["status"] = labels_df.apply(lambda row: (row["vital_status"] == "Dead"), axis=1)

    # Include only the processed columns.
    labels_df = labels_df[["case_id", "status", "status_time"]]

    return labels_df

def merge(labels_df, feature_df):
    merged_df = labels_df.merge(feature_df, on="case_id")

    return merged_df

def get_x_and_y(merged_dataframe):
    # Gets the X and Y matrices for the model to use.
    y_dataframe = merged_dataframe[["status", "status_time"]]
    x_dataframe = merged_dataframe.drop(["case_id", "status", "status_time"], axis=1)

    y_vector = Surv.from_dataframe("status", "status_time", y_dataframe)
    x_matrix = np.asarray(x_dataframe)
    #print(x_matrix, y_vector)
    return x_matrix, y_vector

def split_x_and_Y(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test

def rsf_experiment(X_train, X_test, y_train, y_test):
    rsf = RandomSurvivalForest(n_estimators=50,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           n_jobs=-1,
                           random_state=RANDOM_STATE)
    rsf.fit(X_train, y_train)

    score = rsf.score(X_test, y_test)
    return score


print("-- Reading Data --")
mutation_df, gexp_df, clinical_df = read_data(mutation_tsv, gexp_tsv, clinical_tsv)
labels_df = get_labels(clinical_df)
print (clinical_df)

"""
print("\n###### Mutations Data #######")
merged_df = merge(labels_df, mutation_df)
X, y = get_x_and_y(merged_df)
X_train, X_test, y_train, y_test = split_x_and_Y(X, y)

score = rsf_experiment(X_train, X_test, y_train, y_test)
print("Mutations Concordance Index: ", score)
"""
print("\n###### Gene Expression Data #######")
merged_df = merge(labels_df, gexp_df)
X, y = get_x_and_y(merged_df)
X_train, X_test, y_train, y_test = split_x_and_Y(X, y)

score = rsf_experiment(X_train, X_test, y_train, y_test)
print("Gene Expression Concordance Index: ", score)
