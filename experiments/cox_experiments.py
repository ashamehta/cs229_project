
import cox_regression_main as cox
import feature_selection as fs

import sklearn as sk
from sklearn import model_selection
from sksurv import linear_model as sk_lm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cox_experiment_simple_with_selected_gexp_features(
        train_gexp_df, train_clinical_df, valid_gexp_df, valid_clinical_df, test_gexp_df, test_clinical_df,
        model_df=None, si_tsv=None, num_features=79):
    """
    Train a Cox PH Regression model on |gexp_df| data using top |num_features| features as specified by |model_df|.
    Uses the pre-defined train/valid split to tune the regularization hyperparameter alpha.
    """
    # Select top |num_features| features based on Cox Net results as represented by |model_df|.
    assert model_df is not None or si_tsv is not None
    if model_df is not None:
        print("Selecting a set of %s features from the coxnet results." % num_features)
        selected_train_gexp_df = fs.select_features_from_cox_coef(model_df, train_gexp_df, num_features=num_features)
        selected_valid_gexp_df = fs.select_features_from_cox_coef(model_df, valid_gexp_df, num_features=num_features)
        selected_test_gexp_df = fs.select_features_from_cox_coef(model_df, test_gexp_df, num_features=num_features)
    elif si_tsv is not None:
        print("Selecting a set of %s features from the SI ranking." % num_features)
        selected_train_gexp_df = fs.select_features_from_si_ranking(si_tsv, train_gexp_df, num_features)
        selected_valid_gexp_df = fs.select_features_from_si_ranking(si_tsv, valid_gexp_df, num_features)
        selected_test_gexp_df = fs.select_features_from_si_ranking(si_tsv, test_gexp_df, num_features)

    train_dataset = cox.CoxRegressionDataset(
        selected_train_gexp_df, train_clinical_df, standardize=True, test_size=0.0)
    valid_dataset = cox.CoxRegressionDataset(
        selected_valid_gexp_df, valid_clinical_df, standardize=True, test_size=0.0)
    test_dataset = cox.CoxRegressionDataset(
        selected_test_gexp_df, test_clinical_df, standardize=True, test_size=0.0)

    # Instead of picking arbitrary alphas, select range of alphas based on an initial coxnet run
    small_model = sk_lm.CoxnetSurvivalAnalysis(alpha_min_ratio=0.08, l1_ratio=0.95, max_iter=10000)
    small_model.fit(train_dataset.X, train_dataset.y)
    alphas = list(reversed(list(small_model.alphas_)[0::4]))  # select every 4th alpha in the range.
    print("Selected alphas for this run:", alphas)

    # Fit and evaluate with each model.
    models = [sk_lm.CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=0.95) for a in alphas]
    train_scores, valid_scores = [], []
    for model in models:
        model.fit(train_dataset.X, train_dataset.y)
        train_scores.append(model.score(train_dataset.X, train_dataset.y))
        valid_scores.append(model.score(valid_dataset.X, valid_dataset.y))
    
    # Select model with best alpha coefficient.
    best_model_i = np.argmax(valid_scores)
    best_model = models[best_model_i]

    full_train_X = np.concatenate((train_dataset.X, valid_dataset.X), axis=0)
    full_train_y = np.concatenate((train_dataset.y, valid_dataset.y), axis=0)
    best_model.fit(full_train_X, full_train_y)
    test_score = best_model.score(test_dataset.X, test_dataset.y)

    print("Train scores:", train_scores)
    print("Valid scores:", valid_scores)
    print("Using alpha=%s:\nValidation Score=\t%s\nTraining Score=\t\t\t%s\nTest Score=\t\t\t%s\n" % (
        alphas[best_model_i], valid_scores[best_model_i], train_scores[best_model_i], test_score) )

    return valid_scores[best_model_i], train_scores[best_model_i], test_score, alphas[best_model_i]


def iterative_model_simple_selection(
    train_gexp_df, train_clinical_df, valid_gexp_df, valid_clinical_df, test_gexp_df, test_clinical_df,
    model_df=None, si_tsv=None, log_filename="model_selection_test_log.txt"):
    """
    Trains a Cox PH regression model on |gexp_df| and |clinical_df| for each set of top-N features according to |model_df|.
    Writes results into |log_filename|.
    """
    if model_df is not None:
        num_nonzero_features = np.sum(np.sign(np.abs(np.asarray(model_df))), axis=1)[:70]
    else:
        num_nonzero_features = range(4, 221, 4)

    f = open(log_filename, "a")

    num_selected_features, valid_scores, train_scores, test_scores, alphas = [], [], [], [], []
    for num_f in num_nonzero_features:
        if num_f > 59:
            valid_score, train_score, test_score, alpha = cox_experiment_simple_with_selected_gexp_features(
                train_gexp_df, train_clinical_df, valid_gexp_df, valid_clinical_df, test_gexp_df, test_clinical_df,
                model_df=model_df, si_tsv=si_tsv, num_features=num_f)

            num_selected_features.append(num_f)
            valid_scores.append(valid_score)
            train_scores.append(train_score)
            test_scores.append(test_score)
            alphas.append(alpha)

            f.write("\t".join([str(num_f), str(valid_score), str(train_score), str(test_score), str(alpha)]) + "\n")
            print()

    f.close()

    best_i = np.argmax(valid_scores)
    print("Best model uses %s features." % num_selected_features[best_i])
    print("  Valid Score: %s" % valid_scores[best_i])
    print("  Train Score: %s" % train_scores[best_i])
    print("  Test  Score: %s" % test_scores[best_i])
    print("  Alpha:       %s" % alphas[best_i])

    return


# MAIN RUNS B:
# * Rank features based on training dataset, separate from validation dataset.
# * Select best feature set based on evaluation by validation dataset.
# * Evaluate final model with test dataset.

train_gexp_top05_tsv = "processed_data/gene_expression_top05_train_small.tsv"
valid_gexp_top05_tsv = "processed_data/gene_expression_top05_valid.tsv"
test_gexp_top05_tsv = "processed_data/gene_expression_top05_test.tsv"

train_clinical_tsv = "processed_data/clinical_train_small.tsv"
valid_clinical_tsv = "processed_data/clinical_valid.tsv"
test_clinical_tsv = "processed_data/clinical_test.tsv"

train_gexp_df, train_clinical_df = pd.read_csv(train_gexp_top05_tsv, sep="\t"), pd.read_csv(train_clinical_tsv, sep="\t")
valid_gexp_df, valid_clinical_df = pd.read_csv(valid_gexp_top05_tsv, sep="\t"), pd.read_csv(valid_clinical_tsv, sep="\t")
test_gexp_df, test_clinical_df = pd.read_csv(test_gexp_top05_tsv, sep="\t"), pd.read_csv(test_clinical_tsv, sep="\t")

# model_tsv = "output/cox_model_elastic_gexp_exp8.tsv"
model_tsv = "output/cox_model_elastic_si_gexp_exp2.tsv"
# model_df = pd.read_csv(model_tsv, sep="\t", index_col=0)


# 12 - train_small, coxnet_elast
# 13 - train_small, si -> coxnet_elast
log_file = "cox_output/model_selection_run13_log.txt"

# iterative_model_simple_selection(
#     train_gexp_df, train_clinical_df, valid_gexp_df, valid_clinical_df, test_gexp_df, test_clinical_df,
#     model_df=model_df, si_tsv=None, log_filename=log_file)





def cox_experiment_with_selected_gexp_features(
    train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df,
    model_df=None, si_tsv=None, num_features=79, alpha_score_plot=None):
    """
    Train a Cox PH Regression model on |gexp_df| data using top |num_features| features as specified by |model_df|.
    Uses cross-validation to tune the regularization hyperparameter alpha.
    """
    # Select top |num_features| features based on Cox Net results as represented by |model_df|.
    if model_df is not None:
        print("Selecting a set of %s features from the coxnet results." % num_features)
        selected_train_gexp_df = fs.select_features_from_cox_coef(model_df, train_gexp_df, num_features=num_features)
        selected_test_gexp_df = fs.select_features_from_cox_coef(model_df, test_gexp_df, num_features=num_features)
    elif si_tsv is not None:
        print("Selecting a set of %s features from the SI ranking." % num_features)
        selected_train_gexp_df = fs.select_features_from_si_ranking(si_tsv, train_gexp_df, num_features)
        selected_test_gexp_df = fs.select_features_from_si_ranking(si_tsv, test_gexp_df, num_features)
    else:
        selected_train_gexp_df = train_gexp_df
        selected_test_gexp_df = test_gexp_df
        
    train_dataset = cox.CoxRegressionDataset(
        selected_train_gexp_df, train_clinical_df, standardize=True, test_size=0.0)
    test_dataset = cox.CoxRegressionDataset(
        selected_test_gexp_df, test_clinical_df, standardize=True, test_size=0.0)

    # Instead of picking arbitrary alphas, select range of alphas based on an initial coxnet run
    small_model = sk_lm.CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, l1_ratio=0.95, max_iter=10000)
    small_model.fit(train_dataset.X, train_dataset.y)
    alphas = list(reversed(list(small_model.alphas_)[0::3]))  # select every 4th alpha in the range.
    print("Selected alphas for this run:", alphas)

    # Train models for each of the alpha values. Evaluate via cross-validation.
    models = [sk_lm.CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=0.95, fit_baseline_model=True) for a in alphas]
    results = [model_selection.cross_validate(model, train_dataset.X, train_dataset.y, return_train_score=True) for model in models]

    # Get mean and standard deviation of the scores.
    valid_scores = [np.mean(result["test_score"]) for result in results]
    valid_stds = [np.std(result["test_score"]) for result in results]
    train_scores = [np.mean(result["train_score"]) for result in results]
    train_stds = [np.std(result["train_score"]) for result in results]
    print("Mean Cross-Validation Scores:", valid_scores)
    print("Std Dev Cross-Validation Scores:", valid_stds)

    # Keep track of best score seen.
    max_score, argmax_score = 0, None
    for i, score in enumerate(valid_scores):
        if score > max_score:
            max_score, argmax_score = score, i

    # Test score.
    model = models[argmax_score]
    model.fit(train_dataset.X, train_dataset.y)
    test_score = model.score(test_dataset.X, test_dataset.y)
    # cox.save_cox_model(model, selected_train_gexp_df, output_file="output/cox_model_selected_exp12.tsv")

    # If needed, plot alpha vs validation score
    if alpha_score_plot is not None:
        plt.cla()
        plt.plot(alphas, valid_scores, c="blue")
        plt.plot(alphas, train_scores, c="orange")
        plt.savefig(alpha_score_plot)

    print("Using alpha=%s:\nAverage Cross-Validation Score=\t%s\nTraining Score=\t\t\t%s\nTest Score=\t\t\t%s\n" % (
        alphas[argmax_score], max_score, train_scores[argmax_score], test_score) )

    return max_score, train_scores[argmax_score], test_score, alphas[argmax_score], model


def iterative_model_selection(
    train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df,
    model_df=None, log_filename="model_selection_test_log.txt", si_tsv=None):
    """
    Trains a Cox PH regression model on |gexp_df| and |clinical_df| for each set of top-N features according to |model_df|.
    Writes results into |log_filename|.
    """
    if model_df is not None:
        num_nonzero_features = np.sum(np.sign(np.abs(np.asarray(model_df))), axis=1)[:70]
    else:
        num_nonzero_features = range(4, 221, 4)

    f = open(log_filename, "a")

    ave_cross_val_scores = []
    max_score, max_s = 0, None
    for num_f in num_nonzero_features:
        if num_f > 0:
            valid_score, train_score, test_score, alpha, model = cox_experiment_with_selected_gexp_features(
                train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df, model_df=model_df, si_tsv=si_tsv, num_features=num_f)

            ave_cross_val_scores.append((valid_score, train_score, test_score, alpha))
            if valid_score > max_score:
                max_score, max_s = valid_score, (num_f, valid_score, train_score, test_score, alpha)

            f.write("\t".join([str(num_f), str(valid_score), str(train_score), str(test_score), str(alpha)]) + "\n")
            print()
    f.close()




# MAIN RUNS A:
# * Rank features based on training dataset
# * Select best feature set with cross-validation on proposed feature sets.
# * Evaluate final model with test dataset.

train_gexp_top05_tsv = "processed_data/gene_expression_top05_train.tsv"
test_gexp_top05_tsv = "processed_data/gene_expression_top05_test.tsv"
train_clinical_tsv = "processed_data/clinical_train.tsv"
test_clinical_tsv = "processed_data/clinical_test.tsv"

# train_clinical_df, test_clinical_df = pd.read_csv(train_clinical_tsv, sep="\t"), pd.read_csv(test_clinical_tsv, sep="\t")
# train_gexp_df, test_gexp_df = pd.read_csv(train_gexp_top05_tsv, sep="\t"), pd.read_csv(test_gexp_top05_tsv, sep="\t")


# cox_lasso_gexp = "output/cox_model_lasso_gexp_exp5.tsv"
# model_df = pd.read_csv(cox_lasso_gexp, sep="\t", index_col=0)
# cox_elast_gexp = "output/cox_model_elastic_gexp_exp5.tsv"
# model_df = pd.read_csv(cox_elast_gexp, sep="\t", index_col=0)
cox_elast_si_gexp = "output/cox_model_elastic_si_gexp_exp1.tsv"
# model_df = pd.read_csv(cox_elast_si_gexp, sep="\t", index_col=0)

# 7 - df
# 8 - train1, test1, exp5 (8a = faulty, 8b = corrected+elastic, 8c = corrected+lasso)
# 9 - train2, test2, exp6
# 10 -train3, test3, exp7
# 11 - train1, test1, si_exp1 (11a = si -> coxnet, 11b = si only, 11c = repeat of 11a)
# 12 - see above runs B
# 13 - see above runs B
# cox_experiment_with_selected_gexp_features(train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df, model_df, num_features=101)
# iterative_model_selection(train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df, model_df, "cox_output/model_selection_run11c_log.txt")





# MAIN RUNS C:
# Same as Main Runs A, but with baseline clinical factors.

train_clinical_tsv = "processed_data/clinical_train.tsv"
test_clinical_tsv = "processed_data/clinical_test.tsv"
train_clinical_df, test_clinical_df = pd.read_csv(train_clinical_tsv, sep="\t"), pd.read_csv(test_clinical_tsv, sep="\t")

train_feature_df = train_clinical_df.loc[:, ["case_id", "age_at_diagnosis", "figo_stage"]]
test_feature_df = test_clinical_df.loc[:, ["case_id", "age_at_diagnosis", "figo_stage"]]
print(train_feature_df.shape)
print(test_feature_df.shape)
train_feature_df = train_feature_df.dropna()
test_feature_df = test_feature_df.dropna()
print(train_feature_df.shape)
print(test_feature_df.shape)
# assert False

cox_experiment_with_selected_gexp_features(
    train_feature_df, train_clinical_df, test_feature_df, test_clinical_df)

# train_dataset = cox.CoxRegressionDataset(train_feature_df, train_clinical_df, standardize=False, test_size=0.0)
# test_dataset = cox.CoxRegressionDataset(test_feature_df, test_clinical_df, standardize=False, test_size=0.0)









