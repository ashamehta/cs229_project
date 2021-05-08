
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


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


# plot_num_features_vs_scores("test_log4.txt", plot_filename="post_analysis/num_features_vs_scores_log4.png")
plot_feature_coefficients("output/selected_coxnet_model3.tsv",
                          "post_analysis/selected_coxnet_model3_feature_coefficients_top25_2.png")
