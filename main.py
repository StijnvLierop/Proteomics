from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define body fluids
BODY_FLUIDS = ["saliva", "semen", "vaginalfluid", "urine", "blood"]

# Read data
po_df = pd.read_csv("2581_PureOnly_Report_25MAR24_NP.csv", sep=';')
co_df = pd.read_csv("2581_CombiOnly_Report_25MAR24_NP.csv", sep=';')


def add_protein_count_per_body_fluid(df: pd.DataFrame,
                                     samples_to_exclude: Iterable[str] = None) -> pd.DataFrame:
    # Define samples to look at
    samples = [x for x in df.columns if x.endswith("PG.Quantity")]
    if samples_to_exclude:
        samples = [x for x in samples if x not in samples_to_exclude]

    # For each protein and body fluid
    for key, protein_data in df.iterrows():
        for fluid in BODY_FLUIDS:
            # Initialize count to 0
            count = 0

            # Loop over samples
            for sample in samples:
                # If protein intensity is not NaN, protein is present in sample so add 1 to the count for the current fluid
                if fluid in sample and not np.isnan(float(str(protein_data[sample]).replace(',', '.'))):
                    count += 1

            # Store count for current protein and fluid in dataframe
            df.loc[key, fluid] = count

    return df


def get_discriminative_proteins(df: pd.DataFrame):
    return


if __name__ == '__main__':
    # Set samples to exclude
    samples_to_exclude = None

    # Add nr of times each protein occurs in a body fluid
    po_df = add_protein_count_per_body_fluid(po_df, samples_to_exclude)
    print(po_df.sort_values(by='blood', ascending=False)[['PG.ProteinAccessions', 'blood']])
