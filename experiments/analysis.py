
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import feature_selection as fs
import cox_experiments as cox_exp
import cox_regression_main as cox

def get_times(y_raw):
    event_field, time_field = y_raw.dtype.names
    y_event, y_time = y_raw[event_field], y_raw[time_field]
    return y_time



def plot_survival_curves(surv_fns, y_time, feature_values, plot_filename):
    time_points = np.quantile(y_time, np.linspace(0, 1.0, 100))
    legend_handles = []
    legend_labels = []
    _, ax = plt.subplots(figsize=(9, 6))
    for fn, label in zip(surv_fns, feature_values.astype(int)):
        line, = ax.step(time_points, fn(time_points), where="post",
                       color="C{:d}".format(label), alpha=0.5)
        name = "positive" if label == 1 else "negative"
        if name not in legend_labels:
            legend_labels.append(name)
            legend_handles.append(line)

    ax.legend(legend_handles, legend_labels)
    ax.set_xlabel("time")
    ax.set_ylabel("Survival probability")
    ax.grid(True)
    plt.savefig(plot_filename)
    return


train_gexp_top05_tsv = "processed_data/gene_expression_top05_train.tsv"
test_gexp_top05_tsv = "processed_data/gene_expression_top05_test.tsv"
train_clinical_tsv = "processed_data/clinical_train.tsv"
test_clinical_tsv = "processed_data/clinical_test.tsv"

train_clinical_df, test_clinical_df = pd.read_csv(train_clinical_tsv, sep="\t"), pd.read_csv(test_clinical_tsv, sep="\t")
train_gexp_df, test_gexp_df = pd.read_csv(train_gexp_top05_tsv, sep="\t"), pd.read_csv(test_gexp_top05_tsv, sep="\t")

cox_elast_gexp = "output/cox_model_elastic_gexp_exp5.tsv"
model_df = pd.read_csv(cox_elast_gexp, sep="\t", index_col=0)



# Best Model from CoxNet Feature selection (num_features=97)
num_features = 97
selected_train_gexp_df = fs.select_features_from_cox_coef(
    model_df, train_gexp_df, num_features=num_features)
train_dataset = cox.CoxRegressionDataset(
    selected_train_gexp_df, train_clinical_df, standardize=True, test_size=0.0)

# Get the model.
_, _, _, _, model = cox_exp.cox_experiment_with_selected_gexp_features(
    train_gexp_df, train_clinical_df, test_gexp_df, test_clinical_df, model_df, num_features=num_features)

# Get the survival functions.
surv_fns = model.predict_survival_function(train_dataset.X, alpha=model.alphas[0])

# Get the values of the feature for each sample.
feature_i = list(selected_train_gexp_df.columns).index("ENSG00000204264.7") - 1     # -1 since the "case_id" index is removed.
feature_values = train_dataset.X[:,feature_i]
feature_values = np.clip(np.sign(feature_values), 0, 1)

# Create plot
output_filename = "post_analysis/cox_model_survival_curves_exp5.png"
plot_survival_curves(
    surv_fns, get_times(train_dataset.y), feature_values, plot_filename=output_filename)







def plot_num_features_vs_scores(log_filename, plot_filename="num_features_vs_scores_test.png"):
    stats = pd.read_csv(log_filename, header=None, sep="\t")
    stats = stats.sort_values(0)

    # Plot validation scores
    plt.plot(stats.iloc[:,[0]], stats.iloc[:,[1]], c="blue")

    # Plot training scores
    plt.plot(stats.iloc[:,[0]], stats.iloc[:,[2]], c="orange")

    plt.xlabel("Num top features used in model")
    plt.ylabel("Average Concordance Index Score")
    plt.gca().legend(["Validation", "Training"], loc="lower right")
    plt.savefig(plot_filename)
    plt.show()


def plot_feature_coefficients(model_filename, plot_filename="feature_coefficients.png"):
    model_df = pd.read_csv(model_filename, sep="\t", index_col=0)
    coef_order = model_df.iloc[0].abs().sort_values().index
    model_df = model_df.loc[:, coef_order]

    labels = model_df.columns.tolist()
    x_pos = np.asarray(model_df.iloc[[0],:]).tolist()[0]
    # for i, x in enumerate(x_pos):
    #     if x > 0:
    #         break
    x_pos, labels = x_pos[-25:], labels[-25:]
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(15,10))
    ax.barh(y_pos, x_pos, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient Value")
    plt.savefig(plot_filename)
    plt.show()



# plot_num_features_vs_scores("cox_output/model_selection_run13_log.txt", plot_filename="post_analysis/num_features_vs_scores_log13.png")
# plot_num_features_vs_scores("clustering_analysis/model_selection_si_c1_run1_log.txt", plot_filename="clustering_analysis/num_features_vs_scores_c1_si_run1.png")
plot_feature_coefficients("output/cox_model_selected_exp11.tsv",
                          "post_analysis/selected_coxnet_model3_feature_coefficients_top25_exp11.png")




def get_cox_net_features(cox_net_features_tsv, top_n):
    cox_features = {}
    f = open(cox_net_features_tsv, "r")
    for line in f.readlines():
        terms = line.strip().split("\t")
        feature, rank = terms[0], int(terms[1])
        if rank > top_n:
            print(rank)
            break
        cox_features[feature] = rank
    f.close()
    return cox_features


def compare(cox_net_features_tsv1, cox_net_features_tsv2, top_n):
    pass

# cox_net_features = get_cox_net_features("output/cox_model_top_ranked_features5.tsv", top_n=71)
# cox_si_net_features = get_cox_net_features("output/cox_si_model_top_ranked_features1.tsv", top_n=93)
# print(len(cox_net_features), len(cox_si_net_features))

# gexp_df = pd.read_csv("processed_data/gene_expression_top05_train.tsv", sep="\t")
# cox_net_model_df = pd.read_csv("output/cox_model_elastic_gexp_exp5.tsv", sep="\t", index_col=0)
# cox_si_model_df = pd.read_csv("output/cox_model_elastic_si_gexp_exp1.tsv", sep="\t", index_col=0)
# selected_net_df = fs.select_features_from_cox_coef(cox_net_model_df, gexp_df, num_features=71)
# selected_si_df = fs.select_features_from_cox_coef(cox_si_model_df, gexp_df, num_features=93)

# selected_net_df = selected_net_df.drop(["case_id"], axis=1)
# selected_si_df = selected_si_df.drop(["case_id"], axis=1)
# # print(selected_net_df)
# cox_net_features = set(selected_net_df.columns)
# cox_si_net_features = set(selected_si_df.columns)

# common_features = cox_net_features.intersection(cox_si_net_features)
# cox_net_only = cox_net_features - common_features
# cox_si_only = cox_si_net_features - common_features
# print("Num features only in CoxNet selected set:", len(cox_net_only))
# print("Num features only in CoxNet-SI selected set:", len(cox_si_only))
# print("Num features shared:", len(common_features))

# print(common_features)



