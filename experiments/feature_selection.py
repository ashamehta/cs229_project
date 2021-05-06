
import numpy as np
import sklearn as sk
form sklearn import feature_selection

import matplotlib.pyplot as plt


def remove_low_variance_features(feature_df, quantile=0.85, histogram_name="variance"):
    variances = np.var(np.asarray(feature_df)[:,1:], axis=0)
    quant = np.quantile(variances, quantile)

    plt.hist(variances)
    plt.yscale("log")
    plt.title("Histogram of Feature Variances")
    plt.xlabel("Variance of Feature")
    plt.ylabel("Num features with specified variance (log)")
    plt.savefig(variance + ".png")

    sel = VarianceThreshold(threshold=quant)
    sel.fit(feature_df.drop(columns=["case_id"]))
    retained_indeces = sel.get_support(indices=True)
    new_feature_df = feature_df.iloc[:,retained_indices]

    return new_feature_df



mutation_tsv = "processed_data/mutations_matrix.tsv"
gexp_tsv = "processed_data/gene_expression_matrix.tsv"
clinical_tsv = "processed_data/clinical_processed.tsv"

raw_mut_df = pd.read_csv(mutation_tsv, sep="\t")
mut_df = remove_low_variance_features(raw_mut_df, histogram_name="mutations_variance")

