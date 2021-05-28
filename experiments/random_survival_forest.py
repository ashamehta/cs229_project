import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import uniform, randint, zscore
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import preprocessing
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest


import feature_selection as fs


mutation_tsv = "processed_data/mutations_matrix.tsv"
gexp_tsv = "processed_data/gene_expression_matrix.tsv"
gexp_tsv_variance_03 = "processed_data/gene_expression_top03_matrix.tsv"
gexp_tsv_variance_05 = "processed_data/gene_expression_top05_matrix.tsv"
gexp_tsv_variance_15 = "processed_data/gene_expression_top15_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed.tsv"

RANDOM_STATE = 44

def read_data(mutation_tsv, gexp_tsv, gexp_df_03, gexp_df_05, gexp_df_15, clinical_tsv):
    mut_df = pd.read_csv(mutation_tsv, sep="\t")
    gexp_df = pd.read_csv(gexp_tsv, sep="\t")
    gexp_df_03 = pd.read_csv(gexp_tsv_variance_03, sep="\t")
    gexp_df_05 = pd.read_csv(gexp_tsv_variance_05, sep="\t")
    gexp_df_15 = pd.read_csv(gexp_tsv_variance_15, sep="\t")
    clin_df = pd.read_csv(clinical_tsv, sep="\t")
    return mut_df, gexp_df, gexp_df_03, gexp_df_05, gexp_df_15, clin_df

def standardize_data(X):
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    return X

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

    return x_matrix, y_vector

def split_x_and_Y(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

    return X_train, X_val, X_test, y_train, y_val, y_test

def rsf_experiment(X_train, X_test, y_train, y_test, best_params):
    rsf = RandomSurvivalForest(n_estimators=best_params['n_estimators'],
                           min_samples_split=best_params['min_samples_split'],
                           max_depth=best_params['max_depth'],
                           max_features=best_params['max_features'],
                           min_samples_leaf=best_params['min_samples_leaf'],
                           random_state=RANDOM_STATE)
    rsf = rsf.fit(X_train, y_train)

    score = rsf.score(X_test, y_test)
    return score

def rsf_hyperparameter_random_search(X_train, X_val, y_train, y_val):
    rsf = RandomSurvivalForest(random_state=RANDOM_STATE, n_jobs=4)

    param_distributions = {
        'n_estimators': randint(1, 100),
        'max_depth': uniform(1, 100),
        'min_samples_split': uniform(0.0, 1.0),
        'min_samples_leaf': randint(1, 10),
        'max_features': uniform(0, 1),
    }

    rsf_random_search = RandomizedSearchCV(
        rsf, param_distributions=param_distributions, n_iter=50, n_jobs=-1, cv=3, random_state=RANDOM_STATE)
    #print("best score: ", model_random_search.best_score_)
    tuned_rsf = rsf_random_search.fit(X_train, y_train)

    print(
        f"Validation c index: "
        f"{tuned_rsf.score(X_val, y_val):.3f}")
    print(
        f"The best set of parameters is: {tuned_rsf.best_params_}"
    )

    return tuned_rsf.best_params_

# Read processed data into dataframes
print("-- Reading Data --")
mutation_df, gexp_df, gexp_df_03, gexp_df_05, gexp_df_15, clinical_df = \
    read_data(mutation_tsv, gexp_tsv,gexp_tsv_variance_03, gexp_tsv_variance_05,gexp_tsv_variance_15, clinical_tsv)
labels_df = get_labels(clinical_df)

print("-- Mutations Data --")
merged_df = merge(labels_df, mutation_df)
X, y = get_x_and_y(merged_df)
X_train, X_val, X_test, y_train, y_val, y_test = split_x_and_Y(X, y)

best_params = rsf_hyperparameter_random_search(X_train, X_val, y_train, y_val)
score = rsf_experiment(X_train, X_test, y_train, y_test, best_params)
print("Mutations Concordance Index: ", score)

print("\n###### Gene Expression Data - variance thresholding top 3% #######")
merged_df = merge(labels_df, gexp_df_03)

X, y = get_x_and_y(merged_df)
X = standardize_data(X)
X_train, X_val, X_test, y_train, y_val, y_test = split_x_and_Y(X, y)

best_params = rsf_hyperparameter_random_search(X_train, X_val, y_train, y_val)
score = rsf_experiment(X_train, X_test, y_train, y_test, best_params)
print("97th quantile Gene Expression score:", score)

print("\n###### Gene Expression Data w/ random search & Coxnet #######")
coef_df = pd.read_csv("~/Documents/Github/cs229_project/experiments/output/cox_model_elastic_gexp_exp3.tsv", sep="\t")
gexp_df = fs.select_features_from_cox_coef(coef_df, gexp_df, 10)

# Save scaled gene dataframe to csv
gexp_df.to_csv('gene_expression_top5_paper_normalized', sep="\t", index=False)

merged_df = merge(labels_df, gexp_df)
X, y = get_x_and_y(merged_df)
X = standardize_data(X)

X_train, X_val, X_test, y_train, y_val, y_test = split_x_and_Y(X, y)

best_params = rsf_hyperparameter_random_search(X_train, X_val, y_train, y_val)
score = rsf_experiment(X_train, X_test, y_train, y_test, best_params)
print("Coxnet Gene Expression score : ", score)

print("\n###### Gene Expression Data w/ random search & TAP1, ZFHX4, CXCL9, FBN1, PTGER3 #######")
gexp_df = fs.select_genes_from_paper(gexp_df)

# Save scaled gene dataframe to csv
gexp_df.to_csv('gene_expression_top5_paper.tsv', sep="\t", index=False)

merged_df = merge(labels_df, gexp_df)
X, y = get_x_and_y(merged_df)
X = standardize_data(X)

X_train, X_val, X_test, y_train, y_val, y_test = split_x_and_Y(X, y)

best_params = rsf_hyperparameter_random_search(X_train, X_val, y_train, y_val)
score = rsf_experiment(X_train, X_test, y_train, y_test, best_params)
print("5 genes from paper, Gene Expression score : ", score)

