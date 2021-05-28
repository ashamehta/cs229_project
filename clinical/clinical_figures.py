import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

clinical_tsv = "~/Documents/GitHub/cs229_project/experiments/processed_data/clinical_processed.tsv"

def read_data(clinical_tsv):
    # Read
    clin_df = pd.read_csv(clinical_tsv, sep="\t")
    return clin_df

def main():
    clinical_df = read_data(clinical_tsv)


if __name__ == "__main__":
    main()
