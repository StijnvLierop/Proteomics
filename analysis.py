import pandas as pd
from typing import Iterable, Tuple
import numpy as np
import streamlit as st
from pyparsing import results

from utils import column2fluid, \
    get_sample_columns, pure_is_in_mixture
from constants import BODY_FLUIDS


@st.cache_data
def get_protein_frequency(protein_df: pd.DataFrame) -> pd.DataFrame:
    # Get sample columns
    sample_columns = get_sample_columns(protein_df)

    # For each protein and body fluid
    for key, protein_data in protein_df.iterrows():
        for fluid in BODY_FLUIDS:
            # Initialize counts to 0
            protein_in_sample_count = 0
            total_fluid_samples = 0

            # Loop over samples
            for sample in sample_columns:
                # If fluid sample add 1 to the fluid sample count
                if fluid in sample:
                    total_fluid_samples += 1
                    # If protein is also present in sample add 1 to the count
                    # for the current fluid
                    if bool(protein_data[sample]):
                        protein_in_sample_count += 1

            # Store relative count for current protein and fluid in dataframe
            if total_fluid_samples > 0:
                protein_df.loc[key, fluid] = (protein_in_sample_count
                                              / total_fluid_samples
                                              * 100)
            else:
                protein_df.loc[key, fluid] = np.nan

    return protein_df[['PG.ProteinDescriptions'] + BODY_FLUIDS]


@st.cache_data
def get_protein_intensity(po_protein_df: pd.DataFrame,
                          proteins_per_sample: pd.DataFrame) -> pd.DataFrame:
    # Create new dataframe
    protein_intensity_df = pd.DataFrame(
        columns=['PG.ProteinDescriptions', 'intensity', 'body fluid', 'sample'])

    # Get sample columns
    sample_columns = get_sample_columns(proteins_per_sample)

    # Loop over samples
    for sample in sample_columns:

        # Get body fluid of sample
        fluid = column2fluid(sample)[0]

        # Get proteins in sample
        proteins_in_sample = proteins_per_sample.loc[
            proteins_per_sample[
                sample], 'PG.ProteinDescriptions'].to_list()

        # Get intensities of selected proteins for selected sample
        protein_intensities = po_protein_df.loc[
            po_protein_df['PG.ProteinDescriptions'].isin(proteins_in_sample)][
            ['PG.ProteinDescriptions', sample]]
        if len(protein_intensities) > 0:
            protein_intensities.rename(columns={sample: 'intensity'},
                                       inplace=True)
            protein_intensities['body fluid'] = fluid
            protein_intensities['sample'] = sample

            # Store protein intensities in dataframe
            protein_intensity_df = pd.concat(
                [protein_intensity_df, protein_intensities])

    return protein_intensity_df


@st.cache_data
def add_mean_protein_intensity(protein_freqs: pd.DataFrame,
                               protein_intensities: pd.DataFrame) \
        -> pd.DataFrame:
    # Loop over body fluids
    for fluid in BODY_FLUIDS:

        # Only take into account data of current fluid
        protein_intensities_fluid = protein_intensities[
            protein_intensities['body fluid'] == fluid]

        # Group by protein and take mean over samples
        protein_intensities_mean = \
            protein_intensities_fluid.groupby('PG.ProteinDescriptions')[
                'intensity'].mean(numeric_only=True).reset_index()

        # Add column to dataframe
        protein_freqs[f'mean protein intensity over samples {fluid}'] = np.nan

        # Add to protein_counts
        for key, protein in protein_intensities_mean.iterrows():
            protein_freqs.loc[
                protein_freqs['PG.ProteinDescriptions'] == protein[
                    'PG.ProteinDescriptions'],
                f'mean protein intensity over samples {fluid}'] = \
                protein['intensity']

    return protein_freqs


@st.cache_data
def filter_on_peptide_count(pure_peptide_df: pd.DataFrame,
                            peptide_threshold: int) -> pd.DataFrame:
    # Get sample columns
    sample_columns = get_sample_columns(pure_peptide_df)

    # Raise value error if no sample columns are found
    if len(sample_columns) == 0:
        raise ValueError(
            "No sample columns found. Please ensure the "
            "import file has the correct format.")

    pure_peptide_df_samples = pure_peptide_df[['PG.Genes',
                                               'PG.ProteinAccessions',
                                               'PG.ProteinDescriptions']
                                              + sample_columns]

    # Replace all numbers with 1's and all NaNs with 0's
    pure_peptide_df_samples = pure_peptide_df_samples.fillna(0)
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


@st.cache_data
def add_gini_impurity(protein_frequency: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Calculate gini impurity and add to dataframe
    protein_frequency['gini impurity'] = protein_frequency.apply(
        lambda row: gini_impurity(np.array(row[BODY_FLUIDS])), axis=1)

    # Get proteins that never occur
    proteins_in_no_body_fluids = protein_frequency[
        protein_frequency['gini impurity'].isna()]

    # Filter out proteins that never occur
    protein_frequency = protein_frequency[
        ~protein_frequency['gini impurity'].isna()]

    # Add helper column for sorting
    protein_frequency['max frequency for protein'] = (
        protein_frequency[BODY_FLUIDS].max(axis=1))

    # Sort values in protein count dataframe
    # based on #1 lowest gini impurity
    # and #2 highest relative sample count
    protein_frequency = (protein_frequency.
                         sort_values(by=['gini impurity',
                                         'max frequency for protein'],
                                     ascending=[True, False])
                         )

    return protein_frequency, proteins_in_no_body_fluids


@st.cache_data
def get_identifying_proteins(protein_frequency: pd.DataFrame) \
        -> pd.DataFrame:
    # Check if intensity is already calculated, otherwise do this first
    if not any(
            protein_frequency.columns.str.contains('mean protein intensity')):
        raise ValueError("Protein intensity not found in data. "
                         "Please calculate and add this first.")

    # Create dataframe of fluids mapping to identifying proteins
    identifying_proteins = pd.DataFrame(
        columns=['PG.ProteinDescriptions',
                 'body fluid',
                 '% of samples with this protein',
                 'mean protein intensity over samples']
    )

    # Set dtypes
    identifying_proteins = (identifying_proteins.astype(
        {'PG.ProteinDescriptions': str,
         'body fluid': str,
         '% of samples with this protein': float,
         'mean protein intensity over samples': float}
    )
    )

    # Create a mask of body fluids
    df_fluids = protein_frequency[BODY_FLUIDS]

    # Loop over fluids
    for fluid in BODY_FLUIDS:
        # Get rows where a body fluid is present
        body_fluid_present = df_fluids[fluid] > 0

        # Get rows where no other body fluids are present
        no_other_fluids_present = (df_fluids.drop(fluid, axis=1)
                                   .sum(axis=1) == 0)

        # Get proteins that meet both conditions
        # and store result in dictionary
        identifying_proteins_fluid = protein_frequency.loc[
            body_fluid_present & no_other_fluids_present,
            [
                'PG.ProteinDescriptions',
                fluid,
                f'mean protein intensity over samples {fluid}',
            ]
        ]

        # Transform separately fluid columns to one fluid column
        # and rename original fluid column to relative occurrence
        identifying_proteins_fluid['body fluid'] = fluid
        identifying_proteins_fluid.rename(
            columns={fluid: '% of samples with this protein',
                     f'mean protein intensity over samples {fluid}':
                         'mean protein intensity over samples'},
            inplace=True)
        identifying_proteins = (
            pd.concat([identifying_proteins_fluid
                      .astype(identifying_proteins.dtypes),
                       identifying_proteins
                      .astype(identifying_proteins_fluid.dtypes)]
                      )
        )

    # Sort values in dataframe
    # based on #1 highest relative sample count
    # and #2 highest mean protein intensity over samples
    identifying_proteins.sort_values(by=['% of samples with this protein',
                                         'mean protein intensity over samples'],
                                     ascending=[False, False],
                                     inplace=True)

    return identifying_proteins


@st.cache_data
def pure_mixture_diff(
        proteins_per_pure_sample: pd.DataFrame,
        proteins_per_mixture_sample: pd.DataFrame) \
        -> pd.DataFrame:
    # Create dataframe to store results
    result_df = pd.DataFrame(columns=['PG.ProteinDescriptions',
                                      'body fluid',
                                      'mix sample',
                                      'present in fluid',
                                      'present in mixture'])

    # Get sample columns
    sample_columns_pure = get_sample_columns(proteins_per_pure_sample)
    sample_columns_mixture = get_sample_columns(proteins_per_mixture_sample)

    # Loop over fluids
    for fluid in BODY_FLUIDS:

        # Fluid columns
        sample_columns_pure_fluid = [c for c in sample_columns_pure if
                                     fluid in c]
        sample_columns_mix_fluid = [c for c in sample_columns_mixture if
                                    fluid in c]

        # Filter on fluid
        fluid_pure = proteins_per_pure_sample[
            ['PG.ProteinDescriptions'] + sample_columns_pure_fluid].copy()

        # Get proteins that occurred in the pure samples for this fluid
        proteins_pure = fluid_pure.loc[
            fluid_pure.drop('PG.ProteinDescriptions', axis=1).any(axis=1),
            'PG.ProteinDescriptions'].to_list()

        # Loop over mixtures
        for mixture in sample_columns_mix_fluid:

            # Get proteins in mixture
            proteins_mixture = (proteins_per_mixture_sample.loc[
                proteins_per_mixture_sample[mixture],
                'PG.ProteinDescriptions']
                                .to_list())

            # Get proteins in mixture not in fluid of pure samples
            not_in_pure_fluid = list(set(proteins_mixture) - set(proteins_pure))

            # Get proteins in fluid of pure samples not in mixture
            not_in_mixture = list(set(proteins_pure) - set(proteins_mixture))

            # Add to dataframe
            if len(not_in_pure_fluid) > 0:
                new_df = pd.DataFrame({'PG.ProteinDescriptions':
                                           not_in_pure_fluid})
                new_df['body fluid'] = fluid
                new_df['mix sample'] = mixture
                new_df['present in fluid'] = False
                new_df['present in mixture'] = True
                result_df = pd.concat([result_df, new_df])

            if len(not_in_mixture) > 0:
                new_df = pd.DataFrame({'PG.ProteinDescriptions':
                                           not_in_mixture})
                new_df['body fluid'] = fluid
                new_df['mix sample'] = mixture
                new_df['present in fluid'] = True
                new_df['present in mixture'] = False
                result_df = pd.concat([result_df, new_df])

    return result_df


@st.cache_data
def general_statistics(proteins_per_pure_sample: pd.DataFrame) -> pd.DataFrame:
    # Get sample columns
    sample_columns = get_sample_columns(proteins_per_pure_sample)
    fluids = [column2fluid(x) for x in sample_columns]

    # Nr of samples per body fluid
    fluid_counts = np.unique(fluids, return_counts=True)

    # Create new dataframe with counts
    df = pd.DataFrame()
    df['body fluid'] = fluid_counts[0]
    df['nr of samples'] = fluid_counts[1]

    # Set index to body fluid
    df.set_index(df.columns[0], inplace=True)

    return df


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
