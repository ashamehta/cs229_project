
import cox_regression_main as cox
import feature_selection as fs

import sklearn as sk
from sklearn import model_selection
from sklearn import preprocessing
from sksurv import linear_model as sk_lm
from sksurv.util import Surv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



clinical_tsv = "processed_data/clinical_processed.tsv"
clinical_df = pd.read_csv(clinical_tsv, sep="\t")




print("\n###### Gene Expression Data -- Cox Regression Experiment #######")

print("\n-- 1. Variance Thresholding Feature Selection --")

def variance_threshold(gexp_df, quantile=0.85, output_filename="gene_expression_top15_matrix.tsv"):
    print("Num total features:", gexp_df.shape[1])
    gexp_df2 = fs.remove_low_variance_features(gexp_df, quantile=quantile, features_name="gene_expression")
    print("Num selected features:", gexp_df2.shape[1])
    gexp_df2.to_csv(output_filename, sep="\t", index=False)
    print("Variance-selected feature matrix written to " + output_filename + ".")
    return gexp_df2

gexp_tsv = "processed_data/gene_expression_matrix.tsv"
# gexp_df1 = pd.read_csv(gexp_tsv, sep="\t")
gexp_top05_tsv = "processed_data/gene_expression_top05_matrix.tsv"
# gexp_df2 = variance_threshold(gexp_df1, quantile=0.95, output_filename=gexp_top05_tsv)
gexp_top15_tsv = "processed_data/gene_expression_top15_matrix.tsv"
# gexp_df2 = variance_threshold(gexp_df1, quantile=0.85, output_filename=gexp_top15_tsv)




print("\n-- 2. Feature Ranking based on Coefficients of Lasso-Regularized Cox Regression --")

def coxnet_gexp_experiment(gexp_df, l1_ratio, output_filename="output/cox_model_gexp_exp.tsv"):
    print("Using Ridge Regression for Cox Regression (L1 Regularization) for feature selection.")
    print("  L1 ratio = %s, alpha_min_ratio = 0.01" % l1_ratio)
    dataset = cox.CoxRegressionDataset(gexp_df, clinical_df, standardize=True)
    coxnet_model = sk_lm.CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01)
    cox.basic_train_and_test(dataset, coxnet_model, model_file=output_filename)

exp_num = 3
cox_lasso_gexp = "output/cox_model_lasso_gexp_exp%s.tsv" % exp_num
cox_elast_gexp = "output/cox_model_elastic_gexp_exp%s.tsv" % exp_num
# gexp_df2 = pd.read_csv(gexp_top05_tsv, sep="\t")
# coxnet_gexp_experiment(gexp_df2, 1.0, output_filename=cox_lasso_gexp)
# coxnet_gexp_experiment(gexp_df2, 0.9, output_filename=cox_elast_gexp)


print("\n-- 3. Select a feature set from list provided by previous Cox Regression feature selection process. --")

def cox_experiment_with_selected_gexp_features(gexp_df, model_df, num_features=77, alpha_score_plot=None):
    print("Selecting a set of %s features from the coxnet results." % num_features)
    selected_gexp_df = fs.select_features_from_cox_coef(model_df, gexp_df, num_features=num_features)
    dataset = cox.CoxRegressionDataset(selected_gexp_df, clinical_df, standardize=True)

    # Instead of picking arbitrary alphas, select range of alphas based on an initial coxnet run
    small_model = sk_lm.CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, l1_ratio=0.95, max_iter=10000)
    small_model.fit(dataset.X, dataset.y)
    alphas = list(reversed(list(small_model.alphas_)[0::4]))  # select every 5th alpha in the range.
    print("Selected alphas for this run:", alphas)

    # Train models on the different alpha values. Evaluate via cross-validation.
    models = [sk_lm.CoxnetSurvivalAnalysis(alphas=[a], l1_ratio=0.95) for a in alphas]
    results = [model_selection.cross_validate(model, dataset.X, dataset.y, return_train_score=True) for model in models]

    # Get scores and standard deviations.
    valid_scores = [np.mean(result["test_score"]) for result in results]
    valid_stds = [np.std(result["test_score"]) for result in results]
    train_scores = [np.mean(result["train_score"]) for result in results]
    train_stds = [np.std(result["train_score"]) for result in results]
    print("Mean Cross-Validation Scores:", valid_scores)
    print("Std Dev Cross-Validation Scores:", valid_stds)

    # Record metrics for each model.
    max_score, argmax_score = 0, None
    for i, score in enumerate(valid_scores):
        if score > max_score:
            max_score, argmax_score = score, i

    # Test score.
    model = models[argmax_score]
    model.fit(dataset.X, dataset.y)
    test_score = model.score(dataset.X_test, dataset.y_test)
    # save_cox_model(model, selected_gexp_df, output_file="output/selected_coxnet_model3.tsv")

    # Plot alpha vs validation score
    if alpha_score_plot is not None:
        plt.cla()
        plt.plot(alphas, valid_scores, c="blue")
        plt.plot(alphas, train_scores, c="orange")
        plt.savefig(alpha_score_plot)

    print("Using alpha=%s:\nAverage Cross-Validation Score=\t%s\nTraining Score=\t\t\t%s\nTest Score=\t\t\t%s\n" % (
        alphas[argmax_score], max_score, train_scores[argmax_score], test_score) )
    return max_score, train_scores[argmax_score], test_score, alphas[argmax_score]

gexp_df2 = pd.read_csv(gexp_top05_tsv, sep="\t")
model_df = pd.read_csv(cox_elast_gexp, sep="\t", index_col=0)
# cox_experiment_with_selected_gexp_features(gexp_df2, model_df, num_features=201)


# Iterative Feature Elimination/Addition
def iterative_model_selection(log_filename="model_selection_test_log.txt"):
    num_nonzero_features = np.sum(np.sign(np.abs(np.asarray(model_df))), axis=1)

    f = open(log_filename, "a")
    ave_cross_val_scores = []
    max_score, max_s = 0, None
    for num_f in num_nonzero_features[:70]:
        if num_f > 201:
            valid_score, train_score, test_score, alpha = \
                cox_experiment_with_selected_gexp_features(gexp_df2, model_df, num_features=num_f)
            ave_cross_val_scores.append((valid_score, train_score, test_score, alpha))
            if valid_score > max_score:
                max_score, max_s = valid_score, (num_f, valid_score, train_score, test_score, alpha)
            f.write("\t".join([str(num_f), str(valid_score), str(train_score), str(test_score), str(alpha)]) + "\n")
            print()
    f.close()

iterative_model_selection("model_selection_run1_log.txt")

# print(ave_cross_val_scores)
# print(max_score, max_s)
