import pandas as pd
from typing import Iterable, Tuple
import numpy as np

from utils import get_proteins_per_sample_true_in_mask
from constants import BODY_FLUIDS


def get_protein_count_per_body_fluid(df: pd.DataFrame,
                                     samples_to_exclude: Iterable[str] = None) \
        -> pd.DataFrame:
    # Define samples to look at
    samples = [x for x in df.columns if x.endswith("PEP.Quantity")]

    # Filter on samples to exclude
    if samples_to_exclude:
        samples = [x for x in samples if x not in samples_to_exclude]

    # For each protein and body fluid
    for key, protein_data in df.iterrows():
        for fluid in BODY_FLUIDS:
            # Initialize counts to 0
            protein_in_sample_count = 0
            total_fluid_samples = 0

            # Loop over samples
            for sample in samples:
                # If fluid sample add 1 to the fluid sample count
                if fluid in sample:
                    total_fluid_samples += 1
                    # If protein is also present in sample add 1 to the count
                    # for the current fluid
                    if bool(protein_data[sample]):
                        protein_in_sample_count += 1

            # Store relative count for current protein and fluid in dataframe
            df.loc[key, fluid] = protein_in_sample_count / total_fluid_samples

    return df[['PG.ProteinDescriptions'] + BODY_FLUIDS]


def filter_on_peptide_count(pure_peptide_df: pd.DataFrame,
                            peptide_threshold: int) -> pd.DataFrame:
    # Filter dataframe on samples
    sample_columns = [x for x in pure_peptide_df.columns
                      if x.endswith('PEP.Quantity')]

    if len(sample_columns) == 0:
        raise ValueError(
            "No sample columns found. Please ensure the "
            "import file has the correct format.")

    pure_peptide_df_samples = pure_peptide_df[['PG.Genes',
                                               'PG.ProteinAccessions',
                                               'PG.ProteinDescriptions']
                                              + sample_columns]

    # Replace all numbers with 1's and all NaNs with 0's
    pure_peptide_df_samples.fillna(0, inplace=True)
    for sample in sample_columns:
        pure_peptide_df_samples.loc[
            pure_peptide_df_samples[sample] > 1, sample] = 1

    # Get proteins which have less than peptide_threshold peptides
    proteins_per_sample = (pure_peptide_df_samples.
                           groupby(['PG.Genes',
                                    'PG.ProteinAccessions',
                                    'PG.ProteinDescriptions'])
                           .sum() >= peptide_threshold).reset_index()

    return proteins_per_sample


def add_gini_impurity(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Calculate gini impurity
    df['gini impurity'] = df.apply(
        lambda row: gini_impurity(np.array(row[BODY_FLUIDS])), axis=1)

    # Get proteins that never occur
    proteins_in_no_body_fluids = df[df['gini impurity'].isna()]

    # Filter out proteins that never occur
    df = df[~df['gini impurity'].isna()]

    # Add helper column for sorting
    df['max_relative_sample_count'] = df[BODY_FLUIDS].max(axis=1)

    # Sort values based on #1 lowest gini impurity
    # and #2 highest relative sample count
    df.sort_values(by=['gini impurity', 'max_relative_sample_count'],
                   ascending=[True, False],
                   inplace=True)

    # Drop helper column
    df.drop('max_relative_sample_count', axis=1, inplace=True)

    return df, proteins_in_no_body_fluids


def get_identifying_proteins_per_body_fluid(df: pd.DataFrame) -> pd.DataFrame:
    # Create dataframe of fluids mapping to identifying proteins
    identifying_proteins = pd.DataFrame(columns=['PG.ProteinDescriptions',
                                                 'body fluid',
                                                 'relative occurrence'])
    # Create a mask of body fluids
    df_fluids = df[BODY_FLUIDS]

    # Loop over fluids
    for fluid in BODY_FLUIDS:
        # Get rows where a body fluid is present
        body_fluid_present = df_fluids[fluid] > 0

        # Get rows where no other body fluids are present
        no_other_fluids_present = df_fluids.drop(fluid, axis=1).sum(axis=1) == 0

        # Get proteins that meet both conditions and store result in dictionary
        identifying_proteins_fluid = df.loc[
            body_fluid_present & no_other_fluids_present,
            [
                'PG.ProteinDescriptions',
                fluid]
        ]

        # Transform separately fluid columns to one fluid column
        # and rename original fluid column to relative occurrence
        identifying_proteins_fluid['body fluid'] = fluid
        identifying_proteins_fluid.rename(
            columns={fluid: 'relative occurrence'}, inplace=True)
        identifying_proteins = pd.concat([identifying_proteins_fluid,
                                          identifying_proteins])

    return identifying_proteins


def get_protein_differences_pure_sample_with_mixture(
        proteins_per_pure_sample: pd.DataFrame,
        proteins_per_mixture_sample: pd.DataFrame) -> pd.DataFrame:
    # Loop over body fluids
    for fluid in BODY_FLUIDS:
        # Get pure proteins for this fluid
        fluid_columns_pure = [x for x in proteins_per_pure_sample.columns if
                              x.endswith('PEP.Quantity')]
        pure_proteins_fluid = proteins_per_pure_sample[fluid_columns_pure]

        # Get mixture proteins for this fluid
        fluid_columns_mixture = [x for x in proteins_per_mixture_sample.columns
                                 if x.endswith('PEP.Quantity')]
        mixture_proteins_fluid = proteins_per_mixture_sample[
            fluid_columns_mixture]

        # Store proteins in mixture not in pure in dictionary
        proteins_in_mixture_not_in_pure = get_proteins_per_sample_true_in_mask(
            proteins_per_pure_sample,
            ~pure_proteins_fluid & mixture_proteins_fluid,
            fluid)

        # Store proteins in pure not in mixture in dictionary
        proteins_in_pure_not_in_mixture = get_proteins_per_sample_true_in_mask(
            proteins_per_pure_sample,
            pure_proteins_fluid & ~mixture_proteins_fluid,
            fluid)

    return proteins_in_mixture_not_in_pure, proteins_in_pure_not_in_mixture


def gini_impurity(counts: np.array(int, ndmin=1)) -> float:
    # Get total label count
    sum = np.sum(counts)

    # Return nan if no labels occur
    if sum == 0:
        return np.nan

    # Calculate gi
    probs = np.divide(counts, sum)
    probs_sq = np.square(probs)
    gi = 1 - np.sum(probs_sq)

    return gi
