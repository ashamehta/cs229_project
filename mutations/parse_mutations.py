import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


def analyze_feature_detail(mutations_df):
    # Potential feature set: has_mutation_at_gene_X
    df_by_gene = df.groupby(["Hugo_Symbol"]).size().reset_index(name='Count')

    plt.hist(df_by_gene["Count"].tolist(), bins=50)
    plt.xlabel("Number of cases with mutation at the gene")
    plt.ylabel("Frequency (log)")
    plt.yscale("log")
    plt.show()

    # Potential feature set: (mutation_at_gene_X, variant_type)
    df_by_variant = df.groupby(["Hugo_Symbol", "Variant_Type"]).size().reset_index(name='Count')

    plt.hist(df_by_variant["Count"].tolist(), bins=50)
    plt.xlabel("Number of cases with specific variant at the gene")
    plt.ylabel("Frequency (log)")
    plt.yscale("log")
    plt.show()

    # Potential feature set: (mutation_at_gene_X, variant_type, specific_mutation)
    df_by_mutation = df.groupby(["Hugo_Symbol", "Variant_Type", "Tumor_Seq_Allele2"]).size().reset_index(name='Count')
    # df_selected_mutations = df_by_mutation[df_by_mutation["Count"] > 5]

    plt.hist(df_by_mutation["Count"].tolist(), bins=50)
    plt.yscale("log")
    plt.xlabel("Number of cases with specific mutation")
    plt.ylabel("Frequency (log)")
    plt.show()


def load_mutation_features(mutations_df):
    # Get list of (case, mutated_gene) tuples. Sometimes, multiple types of mutations are detected for the same tuple.
    selected_df = full_df[["Hugo_Symbol", "case_id"]].drop_duplicates()
    print("Number of Unique (Gene, Case) Tuples:", selected_df.shape[0])

    # Define feature vector. The number of features is 16837.
    selected_df_by_gene = selected_df.groupby(["Hugo_Symbol"]).size().reset_index(name='Count')
    selected_df_by_gene.sort_values("Count", inplace=True, ascending=False)
    mut_feature_vec = selected_df_by_gene["Hugo_Symbol"].reset_index(drop=True)
    print("Number of Mutation Features:", len(mut_feature_vec))

    # # Group rows by case_id (the case_id is the case UUID used in the Genome Cancer Atlas).
    # # The number of cases is 436. The maximum number of mutations for a given case is 1428.
    # selected_df_by_case = selected_df.groupby(["case_id"]).size().reset_index(name='Count')
    # selected_df_by_case.sort_values("Count", inplace=True, ascending=False)
    # print(selected_df_by_case)

    def encode_genes(features, genes):
        gene_to_index = {gene: i for i, gene in enumerate(features)}
        return [gene_to_index[gene] for gene in list(genes)]
    # selected_df_by_case = selected_df.groupby(["case_id"])["Hugo_Symbol"].apply(list).reset_index(name="Mutated_Genes")
    selected_df_by_case = selected_df.groupby(["case_id"])["Hugo_Symbol"].apply(
        lambda x: encode_genes(mut_feature_vec, x)
    ).reset_index(name="Mutated_Genes")

    X = sp.lil_matrix((len(selected_df_by_case), len(mut_feature_vec)))
    for row_index, gene_indeces in enumerate(selected_df_by_case["Mutated_Genes"]):
        X[row_index, gene_indeces] = 1

    feature_encoding = {
        "sparse_data": X, "row_to_case_id": selected_df_by_case["case_id"], "col_to_mutation_id": mut_feature_vec
    }
    feature_df = pd.DataFrame.sparse.from_spmatrix(
        feature_encoding["sparse_data"],
        index=feature_encoding["row_to_case_id"], columns=feature_encoding["col_to_mutation_id"]
    )
    feature_encoding["dataframe"] = feature_df
    return feature_encoding


# Read mutation data into a dataframe.
# Data downloaded from https://portal.gdc.cancer.gov/files/51423d79-e9c5-4c4d-b12c-99c1338dbd43
full_df = pd.read_csv("raw_data.tsv", sep="\t")

# Select only the columns most relevant. Documentation at https://docs.gdc.cancer.gov/Data/File_Formats/MAF_Format/
# Can also use "Entrez_Gene_Id" for an integer ID of these genes. Can use "Tumor_Sample_UUID" instead of "case_id"
df = full_df[["Hugo_Symbol", "Variant_Type", "Tumor_Seq_Allele2", "case_id"]]
print("Num cases:", df["case_id"].nunique())
analyze_feature_detail(df)

feature_info = load_mutation_features(df)
feature_info["dataframe"].to_csv("mutations_matrix.tsv", sep="\t")


