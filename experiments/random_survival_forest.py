import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest


mutation_tsv = "processed_data/mutations_matrix.tsv"
gexp_tsv = "processed_data/gene_expression_top15_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed.tsv"

RANDOM_STATE = 44

def read_data(mutation_tsv, gexp_tsv, clinical_tsv):
    # Read
    mut_df = pd.read_csv(mutation_tsv, sep="\t")
    gexp_df = pd.read_csv(gexp_tsv, sep="\t")
    clin_df = pd.read_csv(clinical_tsv, sep="\t")
    return mut_df, gexp_df, clin_df

def get_labels(clinical_df):
    # Get only the relevant columns.
    labels_df = clinical_df[["case_id", "normalized_age_at_index", "vital_status", "days_to_last_follow_up", "days_to_death"]]

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
    labels_df = labels_df[["case_id", "normalized_age_at_index", "status", "status_time"]]

    return labels_df

def merge(labels_df, feature_df):
    merged_df = labels_df.merge(feature_df, on="case_id")
    print(merged_df)

    return merged_df

def get_x_and_y(merged_dataframe):
    # Gets the X and Y matrices for the model to use.
    y_dataframe = merged_dataframe[["status", "status_time"]]
    x_dataframe = merged_dataframe.drop(["case_id", "status", "status_time"], axis=1)

    y_vector = Surv.from_dataframe("status", "status_time", y_dataframe)
    x_matrix = np.asarray(x_dataframe)

    return x_matrix, y_vector

def split_x_and_Y(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test

def rsf_experiment(X_train, X_test, y_train, y_test):
    rsf = RandomSurvivalForest(n_estimators=100,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           random_state=RANDOM_STATE)
    rsf = rsf.fit(X_train, y_train)

    score = rsf.score(X_test, y_test)
    return score


def rsf_experiment_random_search(X_train, X_test, y_train, y_test):
    rsf = RandomSurvivalForest(random_state=RANDOM_STATE)

    # Hyperparameter tuning
    param_distributions = {
        'max_depth': uniform(1, 100),
        'min_samples_leaf': uniform(0, 0.5),
        'max_features': uniform(0, 1),
    }

    rsf_random_search = RandomizedSearchCV(
        rsf, param_distributions=param_distributions, n_iter=50, n_jobs=-1, cv=3, random_state=RANDOM_STATE)
    #print("best score: ", model_random_search.best_score_)
    tuned_rsf = rsf_random_search.fit(X_train, y_train)

    print(
        f"The c-index of random survival forest using a {tuned_rsf.__class__.__name__} is "
        f"{tuned_rsf.score(X_test, y_test):.3f}")
    print(
        f"The best set of parameters is: {tuned_rsf.best_params_}"
    )


print("-- Reading Data --")
mutation_df, gexp_df, clinical_df = read_data(mutation_tsv, gexp_tsv, clinical_tsv)
labels_df = get_labels(clinical_df)


print("\n###### Mutations Data #######")
merged_df = merge(labels_df, mutation_df)
X, y = get_x_and_y(merged_df)
X_train, X_test, y_train, y_test = split_x_and_Y(X, y)

score = rsf_experiment(X_train, X_test, y_train, y_test)
print("Mutations Concordance Index: ", score)

#score_random_search = rsf_experiment_random_search(X_train, X_test, y_train, y_test)
#print("Mutations Concordance Index with Random Search: ", score_random_search)

"""
print("\n###### Gene Expression Data #######")
merged_df = merge(labels_df, gexp_df)
X, y = get_x_and_y(merged_df)
X_train, X_test, y_train, y_test = split_x_and_Y(X, y)

score = rsf_experiment(X_train, X_test, y_train, y_test)
print("Gene Expression Concordance Index: ", score)
"""
