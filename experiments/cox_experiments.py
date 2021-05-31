
import cox_regression_main as cox
import feature_selection as fs

import sklearn as sk
from sklearn import model_selection
from sksurv import linear_model as sk_lm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def cox_experiment_with_selected_gexp_features(
    train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df,
    model_df, num_features=79, alpha_score_plot=None):
    """
    Train a Cox PH Regression model on |gexp_df| data using top |num_features| features as specified by |model_df|.
    Uses cross-validation to tune the regularization hyperparameter alpha.
    """
    # Select top |num_features| features based on Cox Net results as represented by |model_df|.
    print("Selecting a set of %s features from the coxnet results." % num_features)
    selected_train_gexp_df = fs.select_features_from_cox_coef(model_df, train_gexp_df, num_features=num_features)
    train_dataset = cox.CoxRegressionDataset(
        selected_train_gexp_df, train_clinical_df, standardize=True, test_size=0.0)

    selected_test_gexp_df = fs.select_features_from_cox_coef(model_df, test_gexp_df, num_features=num_features)
    test_dataset = cox.CoxRegressionDataset(
        selected_test_gexp_df, test_clinical_df, standardize=True, test_size=0.0)

    # Instead of picking arbitrary alphas, select range of alphas based on an initial coxnet run
    small_model = sk_lm.CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, l1_ratio=0.95, max_iter=10000)
    small_model.fit(train_dataset.X, train_dataset.y)
    alphas = list(reversed(list(small_model.alphas_)[0::3]))  # select every 4th alpha in the range.
    print("Selected alphas for this run:", alphas)

    # Train models for each of the alpha values. Evaluate via cross-validation.
    models = [sk_lm.CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=0.95) for a in alphas]
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
    # save_cox_model(model, selected_gexp_df, output_file="output/selected_coxnet_model3.tsv")

    # If needed, plot alpha vs validation score
    if alpha_score_plot is not None:
        plt.cla()
        plt.plot(alphas, valid_scores, c="blue")
        plt.plot(alphas, train_scores, c="orange")
        plt.savefig(alpha_score_plot)

    print("Using alpha=%s:\nAverage Cross-Validation Score=\t%s\nTraining Score=\t\t\t%s\nTest Score=\t\t\t%s\n" % (
        alphas[argmax_score], max_score, train_scores[argmax_score], test_score) )

    return max_score, train_scores[argmax_score], test_score, alphas[argmax_score]


def iterative_model_selection(
    train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df,
    model_df, log_filename="model_selection_test_log.txt"):
    """
    Trains a Cox PH regression model on |gexp_df| and |clinical_df| for each set of top-N features according to |model_df|.
    Writes results into |log_filename|.
    """
    num_nonzero_features = np.sum(np.sign(np.abs(np.asarray(model_df))), axis=1)

    f = open(log_filename, "a")

    ave_cross_val_scores = []
    max_score, max_s = 0, None
    for num_f in num_nonzero_features[:70]:
        if num_f > 0:
            valid_score, train_score, test_score, alpha = cox_experiment_with_selected_gexp_features(
                train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df, model_df, num_features=num_f)

            ave_cross_val_scores.append((valid_score, train_score, test_score, alpha))
            if valid_score > max_score:
                max_score, max_s = valid_score, (num_f, valid_score, train_score, test_score, alpha)

            f.write("\t".join([str(num_f), str(valid_score), str(train_score), str(test_score), str(alpha)]) + "\n")
            print()
    f.close()




train_gexp_top05_tsv = "processed_data/gene_expression_top05_train.tsv"
test_gexp_top05_tsv = "processed_data/gene_expression_top05_test.tsv"
train_clinical_tsv = "processed_data/clinical_train.tsv"
test_clinical_tsv = "processed_data/clinical_test.tsv"

# train_clinical_df, test_clinical_df = pd.read_csv(train_clinical_tsv, sep="\t"), pd.read_csv(test_clinical_tsv, sep="\t")
# train_gexp_df, test_gexp_df = pd.read_csv(train_gexp_top05_tsv, sep="\t"), pd.read_csv(test_gexp_top05_tsv, sep="\t")


# # cox_lasso_gexp = "output/cox_model_lasso_gexp_exp5.tsv"
# # model_df = pd.read_csv(cox_lasso_gexp, sep="\t", index_col=0)
# cox_elast_gexp = "output/cox_model_elastic_gexp_exp5.tsv"
# model_df = pd.read_csv(cox_elast_gexp, sep="\t", index_col=0)

# 7 - df
# 8 - train1, test1, exp5 (8a = faulty, 8b = corrected+elastic, 8c = corrected+lasso)
# 9 - train2, test2, exp6
# 10 -train3, test3, exp7
# cox_experiment_with_selected_gexp_features(train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df, model_df, num_features=79)
# iterative_model_selection(train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df, model_df, "model_selection_run8b_log.txt")
