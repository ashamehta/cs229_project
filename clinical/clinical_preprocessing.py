import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_tsv(tsv_path):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to TSV file containing dataset.


    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    # Load headers
    tsv_file = open(tsv_path)
    df = pd.read_csv(tsv_file, delimiter="\t", index_col=False)
    return df


def main():
    # load file
    clinical_df = load_tsv('clinical.tsv') #remove index column
    clinical_df = clinical_df.drop_duplicates('case_id')

    # save csv with relevant clinical data indexed by case ID
    clinical_df = clinical_df[['case_id', 'primary_diagnosis', 'figo_stage', 'age_at_index', 'age_at_diagnosis',
                               'days_to_birth', 'days_to_last_follow_up', 'days_to_death', 'race', 'vital_status']]
    clinical_df['age_at_index'] = pd.to_numeric(clinical_df['age_at_index'], errors='coerce')
    clinical_df['age_at_diagnosis'] = pd.to_numeric(clinical_df['age_at_diagnosis'], errors='coerce')
    clinical_df['days_to_birth'] = pd.to_numeric(clinical_df['days_to_birth'], errors='coerce')
    clinical_df['days_to_last_follow_up'] = pd.to_numeric(clinical_df['days_to_last_follow_up'], errors='coerce')
    clinical_df['days_to_death'] = pd.to_numeric(clinical_df['days_to_death'], errors='coerce')

    #convert figo_stage to numerical value
    clinical_df['figo_stage'] = clinical_df['figo_stage'].replace({'Stage IA': 1, 'Stage IB': 1, 'Stage IC': 1,
                                                                   'Stage IIA': 2, 'Stage IIB': 2, 'Stage IIC': 2,
                                                                   'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
                                                                   'Stage IV': 4, "'--": np.nan})
    clinical_df = clinical_df.dropna(axis=0, subset=['figo_stage'])

    #min-max normalized age to produce value between 0 and 1
    normalized_age = (clinical_df['age_at_index']-clinical_df['age_at_index'].min())/(clinical_df['age_at_index'].max()-clinical_df['age_at_index'].min())
    clinical_df.insert(4, 'normalized_age_at_index', normalized_age)

    print('years to death mean', clinical_df['days_to_death'].mean()/365)
    print('years to death SD', clinical_df['days_to_death'].std() / 365)
    print('days to last follow up mean', clinical_df['days_to_last_follow_up'].mean()/365)
    print('days to last follow up SD', clinical_df['days_to_last_follow_up'].std() / 365)

    clinical_df.to_csv('clinical_processed.tsv', index=False, sep="\t", quoting=csv.QUOTE_NONE)

    # clinical_df['age_at_index'] = pd.cut(clinical_df['age_at_index'], bins=[20, 30, 40, 50, 60, 70, 80])
    # survived_5_yrs = clinical_df[clinical_df['days_to_last_follow_up'] >= 1825 and clinical_df['vital_status'] == 'Alive']
    # print(len(survived_5_yrs))

    """
    # plot vital status bucketed by age at index (which is age at diagnosis in years)
    x = clinical_df.loc[clinical_df['vital_status'] == 'Alive', 'age_at_index']
    y = clinical_df.loc[clinical_df['vital_status'] == 'Dead', 'age_at_index']
    plt.hist(x, alpha=0.5, label='alive')
    plt.hist(y, alpha=0.5, label='dead')
    plt.legend(loc='upper right')
    plt.xlabel('age')
    plt.ylabel('frequency')
    plt.savefig('vital_status_by_age_hist.png')
    plt.show()

    # plot vital status bucketed by figo stage (which is age at diagnosis in years)
    x = clinical_df.loc[clinical_df['vital_status'] == 'Alive', 'figo_stage']
    y = clinical_df.loc[clinical_df['vital_status'] == 'Dead', 'figo_stage']
    plt.hist(x, alpha=0.5, label='alive')
    plt.hist(y, alpha=0.5, label='dead')
    plt.legend(loc='upper right')
    plt.xlabel('age')
    plt.ylabel('frequency')
    plt.savefig('vital_status_by_figo_hist.png')
    plt.show()

    # histogram of days to death
    plt.hist(clinical_df['days_to_death']/12)
    plt.xlabel('months to death')
    plt.ylabel('frequency')
    plt.savefig('days_to_death_hist.png')
    plt.show()

    # histogram of age at diagnosis
    avg_age = clinical_df['age_at_index'].mean() #average age = 59.66 yrs
    print(avg_age)
    plt.hist(clinical_df['age_at_index'])
    plt.xlabel('age')
    plt.ylabel('frequency')
    plt.savefig('age_at_diagnosis_hist.png')
    plt.show()
    """

if __name__ == "__main__":
    main()
