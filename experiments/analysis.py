
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# import feature_selection as fs


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



# plot_num_features_vs_scores("cox_output/model_selection_run11b_log.txt", plot_filename="post_analysis/num_features_vs_scores_log11b.png")
# plot_feature_coefficients("output/selected_coxnet_model3.tsv",
#                           "post_analysis/selected_coxnet_model3_feature_coefficients_top25_2.png")




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



