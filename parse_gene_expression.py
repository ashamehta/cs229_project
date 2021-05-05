import argparse

import gzip
import pandas as pd

from scipy import sparse as sp
import numpy as np

import matplotlib.pyplot as plt


class GeneExpressionData(object):
    """
    Class for processing gene expression data into
    """

    def __init__(self, samples_file, data_directory, clinical_file, data_format="FPKM"):
        self.samples_file = samples_file
        self.data_directory = data_directory
        self.clinical_file = clinical_file
        self.data_format = data_format
        self.features = None

    def process_gene_expression_data(self, samples_file):
        # The samples file lists each mRNA data file of this selected dataset.
        f = open(samples_file, "r")
        header = f.readline()

        count, num_files_read = 0, 0
        feature_df = None
        file_suffix = self.data_format + ".txt.gz"

        for line in f.readlines():
            count += 1
            if count % 100 == 0:
                print("Num files scanned:", count)

            # Consider each file, but only process the FPKM files.
            terms = line.strip().split("\t")
            file_id, file_name, case_id = terms[0], terms[1], terms[5]
            if file_name[-1*len(file_suffix):] != file_suffix:
                continue

            # Read the gene expression data into a dataframe.
            gz_file = "/".join([self.data_directory, file_id, file_name])
            df = pd.read_csv(gz_file, compression="gzip", sep="\t", names=["Transcript_Id", case_id])
            sample_df = df[[case_id]].rename(df["Transcript_Id"]).transpose()

            # Update the feature dataframe.
            if feature_df is None:
                feature_df = sample_df
            else:
                feature_df = feature_df.append(sample_df)
            num_files_read += 1

        f.close()
        return feature_df

    def trim_features(self, dataframe):
        # Remove features where the magnitude equals 0 for all samples.
        return dataframe.loc[:, (dataframe != 0).any(axis=0)]

    def assign_case_uuid(self, feature_df, clinical_file):
        # Join feature dataframe and the (case_submitter_id, case_id) dataframe.
        feature_df = feature_df.rename_axis("case_submitter_id").reset_index()
        clinical_df = pd.read_csv(clinical_file, sep="\t", usecols=["case_id", "case_submitter_id"]).drop_duplicates()
        new_df = clinical_df.merge(feature_df, left_on="case_submitter_id", right_on="case_submitter_id")

        # Remove unnecessary "case_submitter_id" column.
        column_names = new_df.columns.tolist()
        column_names.remove("case_submitter_id")
        new_df = new_df[column_names]

        return new_df

    def run_pipeline(self):
        feature_df = self.process_gene_expression_data(self.samples_file)
        feature_df = self.trim_features(feature_df)
        feature_df = self.assign_case_uuid(feature_df, self.clinical_file)
        self.features = feature_df
        return self.features

    def write_features_to_tsv(self, output_file):
        self.features.to_csv(output_file, sep="\t", index=False)



def analyze_feature_landscape(dataframe):
    # For each transcript, get the number of samples where expression levels > 0.
    counts = np.sum(np.sign(df), axis=0)
    plt.hist(counts, bins=100)
    plt.xlabel("Num Samples with Transcript Expressed")
    plt.ylabel("Frequency of Transcripts")
    plt.title("Transcripts binned by number of samples expressing transcript")
    # plt.show()
    plt.savefig("transcripts_per_sample.png")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file", type=str, default="gdc_sample_sheet.2021-05-04.tsv",
                        help="A TSV file listing the transcriptome data files.")
    parser.add_argument("--path_to_data", type=str, default="gene_expression",
                        help="Filepath to the data files specified in samples_file.")
    parser.add_argument("--clinical_file", type=str, default="clinical.tsv",
                        help="A TSV file listing the clinical case data.")
    parser.add_argument("--output_file", type=str, default="gene_expression_matrix.tsv",
                        help="The filepath of an output file to save data to.")
    args = parser.parse_args()

    gene_expression_data = GeneExpressionData(args.samples_file, args.path_to_data, args.clinical_file)
    features = gene_expression_data.run_pipeline()
    print("Num cases, Num features:", features.shape)
    gene_expression_data.write_features_to_tsv(args.output_file)

    # To read the data into a dataframe:
    # df = pd.read_csv(output_file, sep="\t")

main()



