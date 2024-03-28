from typing import Iterable
import numpy as np
import pandas as pd
import streamlit as st

# Define body fluid names
BODY_FLUIDS = ["saliva", "semen", "vaginalfluid", "urine", "blood"]


def add_protein_count_per_body_fluid(df: pd.DataFrame,
                                     samples_to_exclude: Iterable[str] = None)\
                                     -> pd.DataFrame:
    # Define samples to look at
    samples = [x for x in df.columns if x.endswith("PEP.Quantity")]

    # Filter on samples to exclude
    if samples_to_exclude:
        samples = [x for x in samples if x not in samples_to_exclude]

    # For each protein and body fluid
    for key, protein_data in df.iterrows():
        for fluid in BODY_FLUIDS:
            # Initialize count to 0
            count = 0

            # Loop over samples
            for sample in samples:
                # If protein intensity is not NaN, protein is present in
                # sample so add 1 to the count for the current fluid
                if fluid in sample and protein_data[sample]:
                    count += 1

            # Store count for current protein and fluid in dataframe
            df.loc[key, fluid] = count

    return df


def filter_on_peptide_count(pure_peptide_df: pd.DataFrame,
                            peptide_threshold: int) -> pd.DataFrame:
    # Filter dataframe on samples
    sample_columns = [x for x in pure_peptide_df.columns
                      if x.endswith('PEP.Quantity')]
    pure_peptide_df_samples = pure_peptide_df[['PG.Genes',
                                               'PG.ProteinAccessions',
                                               'PG.ProteinDescriptions']
                                              + sample_columns]

    # Get proteins which have less than peptide_threshold peptides
    proteins_per_sample = (pure_peptide_df_samples.
                           groupby(['PG.Genes',
                                    'PG.ProteinAccessions',
                                    'PG.ProteinDescriptions'])
                           .count() >= peptide_threshold).reset_index()

    return proteins_per_sample


if __name__ == '__main__':
    # Set samples to exclude
    samples_to_exclude = None

    # Upload file
    st.header("Upload bestanden")
    pure_file = st.file_uploader(label="PureOnly bestand",
                                 type='.xlsx')

    # Peptide threshold
    peptide_threshold = st.number_input("Peptide threshold (>=)", value=3)

    # When file uploaded
    if pure_file is not None:
        # Read into dataframe
        pure_protein_df = pd.read_excel(pure_file,
                                        sheet_name='2581_PureOnly_Protein')
        pure_peptide_df = pd.read_excel(pure_file,
                                        sheet_name='2581_PureOnly_Peptide')

        # Filter on proteins that have at least n detected peptides per sample
        proteins_per_sample = filter_on_peptide_count(pure_peptide_df,
                                                      peptide_threshold=
                                                      peptide_threshold)

        # Add nr of times each protein occurs in a body fluid
        proteins_per_sample = (
            add_protein_count_per_body_fluid(proteins_per_sample,
                                             samples_to_exclude)
        )

        # Show dataframe
        show_df = proteins_per_sample[['PG.ProteinDescriptions'] + BODY_FLUIDS]
        st.write("Nr of samples with protein per body fluid")
        st.write(show_df)
