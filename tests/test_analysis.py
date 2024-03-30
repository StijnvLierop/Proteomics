import pytest
import numpy as np
import pandas as pd

from analysis import (gini_impurity, filter_on_peptide_count, \
    get_protein_frequency, get_protein_intensity, pure_mixture_diff,
                      get_identifying_proteins, add_mean_protein_intensity)
from utils import preprocess_df


def test_peptide_filter(pure_only_file):
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    po_pep = preprocess_df(po_pep)
    proteins = filter_on_peptide_count(po_pep, 3)
    assert proteins.select_dtypes(bool).loc[0].sum() == 2

    po_pep = preprocess_df(po_pep)
    proteins = filter_on_peptide_count(po_pep, 10)
    assert proteins.select_dtypes(bool).loc[0].sum() == 0


def test_get_protein_frequency(pure_only_file):
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    po_pep = preprocess_df(po_pep)
    proteins = filter_on_peptide_count(po_pep, 3)
    protein_frequency = get_protein_frequency(proteins)
    assert len(protein_frequency) == 1
    assert protein_frequency['saliva'].to_list() == [100]
    assert protein_frequency['vaginalfluid'].to_list() == [100]
    assert protein_frequency['blood'].to_list() == [0]


def test_mean_protein_intensity(pure_only_file):
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    po_pep = preprocess_df(po_pep)
    po_prot = pd.read_excel(pure_only_file,
                            sheet_name='2581_PureOnly_Protein')
    po_prot = preprocess_df(po_prot)
    proteins = filter_on_peptide_count(po_pep, 3)

    intensities = get_protein_intensity(po_prot, proteins)
    assert intensities.iloc[0]['intensity'] == pytest.approx(171176.75)
    assert intensities.iloc[1]['intensity'] == pytest.approx(169310.4063)
    assert intensities.iloc[0]['body fluid'] == "saliva"
    assert intensities.iloc[1]['body fluid'] == "vaginalfluid"
    assert (intensities.iloc[0]['sample'] ==
            "[1] 2581_X_ThomasShehata_14MAR24_DIA_1_saliva1_sample")
    assert (intensities.iloc[1]['sample'] ==
            "[37] 2581_X_ThomasShehata_14MAR24_DIA_63_vaginalfluid15_sample")


def test_add_mean_protein_intensity(pure_only_file):
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    po_pep = preprocess_df(po_pep)
    po_prot = pd.read_excel(pure_only_file,
                            sheet_name='2581_PureOnly_Protein')
    po_prot = preprocess_df(po_prot)
    proteins = filter_on_peptide_count(po_pep, 3)
    intensities = get_protein_intensity(po_prot, proteins)
    protein_freq = get_protein_frequency(proteins)
    protein_freq = add_mean_protein_intensity(protein_freq, intensities)
    assert protein_freq['mean protein intensity over samples saliva'].to_list() == [pytest.approx(171176.75)]
    assert protein_freq['mean protein intensity over samples vaginalfluid'].to_list() == [pytest.approx(169310.4063)]

def test_identifying_proteins(pure_only_file):
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    po_pep = preprocess_df(po_pep)
    po_prot = pd.read_excel(pure_only_file,
                            sheet_name='2581_PureOnly_Protein')
    po_prot = preprocess_df(po_prot)
    proteins = filter_on_peptide_count(po_pep, 3)
    intensities = get_protein_intensity(po_prot, proteins)
    protein_freq = get_protein_frequency(proteins)
    protein_freq = add_mean_protein_intensity(protein_freq, intensities)
    id_proteins = get_identifying_proteins(protein_freq)
    assert len(id_proteins) == 0

    protein_freq['saliva'] = 0
    protein_freq['mean protein intensity over samples saliva'] = np.nan
    id_proteins = get_identifying_proteins(protein_freq)
    assert len(id_proteins) == 1

def test_pure_mix_diff(pure_only_file, combi_only_file):
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    po_pep = preprocess_df(po_pep)
    po_pep = filter_on_peptide_count(po_pep, 3)

    co_pep = pd.read_excel(combi_only_file,
                           sheet_name='2581_CombiOnly_Peptide')
    co_pep = preprocess_df(co_pep)
    co_pep = filter_on_peptide_count(co_pep, 3)

    results = pure_mixture_diff(po_pep, co_pep)
    assert len(results[results['present in pure'] &
                       ~results['present in mixture']]) == 3
    assert len(results[~results['present in pure'] &
                       results['present in mixture']]) == 4

def test_gini_impurity():
    values = np.array([5, 5])
    gi = gini_impurity(values)
    assert gi == 0.5

    values = np.array([0, 10])
    gi = gini_impurity(values)
    assert gi == 0

    values = np.array([3, 7])
    gi = gini_impurity(values)
    assert gi == pytest.approx(0.42)

    values = np.array([0, 0])
    gi = gini_impurity(values)
    assert np.isnan(gi)
