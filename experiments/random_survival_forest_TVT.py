import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import uniform, randint, zscore
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import preprocessing
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest


import feature_selection as fs

clinical_tsv = "processed_data/clinical_processed.tsv"
mutation_tsv = "processed_data/mutations_matrix.tsv"
coef_tsv = "~/Documents/Github/cs229_project/experiments/output/cox_model_elastic_gexp_exp3.tsv"
gexp_tsv = "processed_data/gene_expression_matrix.tsv"
gexp_tsv_variance_03 = "processed_data/gene_expression_top03_matrix.tsv"
#gexp_tsv_variance_05 = "processed_data/gene_expression_top05_matrix.tsv"
#gexp_tsv_variance_15 = "processed_data/gene_expression_top15_matrix.tsv"


RANDOM_STATE = 44

def read_data(clinical_tsv, mutation_tsv, gexp_tsv, coef_tsv, gexp_tsv_variance_03):
    clin_df = pd.read_csv(clinical_tsv, sep="\t")
    mut_df = pd.read_csv(mutation_tsv, sep="\t")
    gexp_df = pd.read_csv(gexp_tsv, sep="\t")
    coef_df = pd.read_csv(coef_tsv, sep="\t")
    gexp_df_03 = pd.read_csv(gexp_tsv_variance_03, sep="\t")
    # gexp_df_05 = pd.read_csv(gexp_tsv_variance_05, sep="\t")
    # gexp_df_15 = pd.read_csv(gexp_tsv_variance_15, sep="\t")
    return clin_df, mut_df, gexp_df, coef_df, gexp_df_03
    # gexp_df_05, gexp_df_15,

def get_labels(clinical_df):
    # Get only the relevant columns.
    labels_df = clinical_df[["case_id", "vital_status", "days_to_last_follow_up", "days_to_death"]]

    # Construct STATUS column for the RSF model to use.
    # The label for each sample should be a structured array (status, time of event or time of censoring).
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

def get_x_and_y(labels_df):
    # Gets the X and Y matrices for the model to use.
    y_dataframe = labels_df[["status", "status_time"]]
    x_dataframe = labels_df.drop(["status", "status_time"], axis=1)

    y_vector = Surv.from_dataframe("status", "status_time", y_dataframe)
    #x_case_id_matrix = np.asarray(x_dataframe)
    return x_dataframe, y_vector

def split_x_and_Y_id(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

    return X_train, X_val, X_test, y_train, y_val, y_test

def feature_train_val_test(X_train_id, X_val_id, X_test_id, feature_df):
    X_train = feature_df[feature_df.case_id.isin(X_train_id.case_id)]
    X_val = feature_df[feature_df.case_id.isin(X_val_id.case_id)]
    X_test = feature_df[feature_df.case_id.isin(X_test_id.case_id)]

    return X_train, X_val, X_test

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

def rsf_hyperparameter_random_search(X_val, y_val):
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
    tuned_rsf = rsf_random_search.fit(X_val, y_val)

    print(
        f"The best set of parameters is: {tuned_rsf.best_params_}"
    )

    return tuned_rsf.best_params_



# Read processed data into dataframes
print("-- Reading Data --")
clinical_df, mutation_df, gexp_df, coef_df, gexp_df_03 = read_data(clinical_tsv, mutation_tsv, gexp_tsv, coef_tsv, gexp_tsv_variance_03)
gexp_top5_df = fs.select_genes_from_paper(gexp_df)
# clinical = 575 samples
# mutation = 436 samples
# gexp = 451 samples
# coef = 100 samples
# gexp and clinical = 376

# Use only data that has gexp data
clinical_gexp_df = clinical_df.merge(gexp_df, how='inner', on='case_id')
clinical_gexp_df = clinical_gexp_df[["case_id", "vital_status", "days_to_last_follow_up", "days_to_death",
                                     "figo_stage", "age_at_index", "normalized_age_at_index"]]
labels_df = get_labels(clinical_gexp_df)

print("-- Get X and y --")
X, y = get_x_and_y(labels_df)

print("-- Split X and y --")

X_train_id, X_val_id, X_test_id, y_train, y_val, y_test = split_x_and_Y_id(X, y)

print("-- TODO Characterize clinical data that has associated gene expression data--")
# Training set
# age_mean_train = clinical_gexp_df[['age_at_index']].mean(axis=1)
# age_std_train = clinical_gexp_df[['age_at_index']].std(axis=1)

# Validation set

# Test set

print("-- Train RSF with Clinical Data that Has Associated Expression Data --")
clinical_baseline_df = clinical_gexp_df[['case_id', 'figo_stage', 'normalized_age_at_index']]
X_train_baseline, X_val_baseline, X_test_baseline = \
    feature_train_val_test(X_train_id, X_val_id, X_test_id, clinical_baseline_df)
X_train_baseline = X_train_baseline.iloc[:, 1:]
X_val_baseline = X_val_baseline.iloc[:, 1:]
X_test_baseline = X_test_baseline.iloc[:, 1:]

best_params = rsf_hyperparameter_random_search(X_val_baseline, y_val)
score = rsf_experiment(X_train_baseline, X_test_baseline, y_train, y_test, best_params)
print("Age and Stage Baseline (w/ GE data):", score)

print("-- Train RSF with Mutations Data, No Feature Selection --")
clinical_mut_df = clinical_df.merge(mutation_df, how='inner', on='case_id')
clinical_mut_df = clinical_mut_df[["case_id", "vital_status", "days_to_last_follow_up", "days_to_death"]]
labels_df = get_labels(clinical_mut_df)
X, y = get_x_and_y(labels_df)

X_train_id, X_val_id, X_test_id, y_train, y_val, y_test = split_x_and_Y_id(X, y)

X_train_mut, X_val_mut, X_test_mut = feature_train_val_test(X_train_id, X_val_id, X_test_id, mutation_df)
X_train_mut = X_train_mut.iloc[:, 1:]
X_val_mut = X_val_mut.iloc[:, 1:]
X_test_mut = X_test_mut.iloc[:, 1:]

best_params = rsf_hyperparameter_random_search(X_val_mut, y_val)
score = rsf_experiment(X_train_mut, X_test_mut, y_train, y_test, best_params)
print("Mutations Concordance Index: ", score)

print("\n###### Gene Expression Data - variance thresholding top 3% #######")
X_train_variance, X_val_variance, X_test_variance = feature_train_val_test(X_train_id, X_val_id, X_test_id, gexp_df_03)
X_train_variance = X_train_variance.iloc[:, 1:]
X_val_variance = X_val_variance.iloc[:, 1:]
X_test_variance = X_test_variance.iloc[:, 1:]

best_params = rsf_hyperparameter_random_search(X_val_variance, y_val)
score = rsf_experiment(X_train_variance, X_test_variance, y_train, y_test, best_params)
print("97th quantile Gene Expression score:", score)

print("\n###### Gene Expression Data w/ random search & Coxnet #######")
all_coef_df = pd.read_csv("~/Documents/Github/cs229_project/experiments/output/cox_model_elastic_gexp_exp3.tsv", sep="\t")
coef_df = fs.select_features_from_cox_coef(all_coef_df, gexp_df, 10)

X_train_coxnet, X_val_coxnet, X_test_coxnet = feature_train_val_test(X_train_id, X_val_id, X_test_id, coef_df)
X_train_coxnet = X_train_coxnet.iloc[:, 1:]
X_val_coxnet = X_val_coxnet.iloc[:, 1:]
X_test_coxnet = X_test_coxnet.iloc[:, 1:]

best_params = rsf_hyperparameter_random_search(X_val_coxnet, y_val)
score = rsf_experiment(X_train_coxnet, X_test_coxnet, y_train, y_test, best_params)
print("Coxnet Gene Expression score : ", score)

print("\n###### Gene Expression Data w/ random search & TAP1, ZFHX4, CXCL9, FBN1, PTGER3 #######")
X_train_top5, X_val_top5, X_test_top5 = feature_train_val_test(X_train_id, X_val_id, X_test_id, gexp_top5_df)
X_train_top5 = X_train_top5.iloc[:, 1:]
X_val_top5 = X_val_top5.iloc[:, 1:]
X_test_top5 = X_test_top5.iloc[:, 1:]

best_params = rsf_hyperparameter_random_search(X_val_top5, y_val)
score = rsf_experiment(X_train_top5, X_test_top5, y_train, y_test, best_params)
print("5 genes from paper, Gene Expression score : ", score)
