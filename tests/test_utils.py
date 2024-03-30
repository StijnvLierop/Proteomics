import numpy as np
import pandas as pd
from conftest import pure_only_file, combi_only_file

from utils import (get_sample_columns, exclude_samples,
                   column2fluid, preprocess_df, pure_is_in_mixture)


def test_exclude_samples(pure_only_file):
    samples_to_exclude = ['[1] 2581_X_ThomasShehata_14MAR24_DIA_1'
                          '_saliva1.raw.PEP.Quantity']
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    new_df = exclude_samples(po_pep, samples_to_exclude)
    columns_before_exclusion = po_pep.columns.tolist()
    columns_after_exclusion = new_df.columns.tolist()
    assert columns_before_exclusion != columns_after_exclusion

    diff = set(columns_before_exclusion) - set(columns_after_exclusion)
    assert list(diff)[0] == samples_to_exclude[0]


def test_exclude_samples_sample_does_not_exist(pure_only_file):
    samples_to_exclude = ['[1] 2581_X_ThomasShehata_14MAR24_DIA_1_'
                          'saliva1.raw.PEP.Quantity']
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    po_pep = preprocess_df(po_pep)
    new_df = exclude_samples(po_pep, samples_to_exclude)
    columns_before_exclusion = po_pep.columns.tolist()
    columns_after_exclusion = new_df.columns.tolist()
    assert columns_before_exclusion == columns_after_exclusion


def test_column2fluid():
    column = '[18] 2581_X_ThomasShehata_14MAR24_DIA_8_urine11.raw.PG.Quantity'
    fluid = column2fluid(column)
    assert fluid == 'urine'


def test_preprocess_dataframe(pure_only_file, combi_only_file):
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    preprocessed = preprocess_df(po_pep)
    assert len(po_pep.columns) == len(preprocessed.columns)
    sample_columns = [x for x in po_pep if x.endswith('_sample')]
    assert len(sample_columns) == 4

    co_pep = pd.read_excel(combi_only_file,
                           sheet_name='2581_CombiOnly_Peptide')
    preprocessed = preprocess_df(co_pep)
    assert len(co_pep.columns) == len(preprocessed.columns)
    sample_columns = [x for x in co_pep if x.endswith('_sample')]
    assert len(sample_columns) == 32


def test_get_sample_columns(pure_only_file):
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    po_pep = preprocess_df(po_pep)
    po_pep_sample_columns = get_sample_columns(po_pep)
    true_po_pep_sample_columns = ["[1] 2581_X_ThomasShehata_14MAR24"
                                  "_DIA_1_saliva1_sample",
                                  "[8] 2581_X_ThomasShehata_14MAR24"
                                  "_DIA_3_blood10_sample",
                                  "[37] 2581_X_ThomasShehata_14MAR24"
                                  "_DIA_63_vaginalfluid15_sample",
                                  "[38] 2581_X_ThomasShehata_14MAR24"
                                  "_DIA_11_blood13_sample"]
    assert po_pep_sample_columns == true_po_pep_sample_columns


def test_pure_is_in_mixture():
    sample_name_pure = "[1] 2581_X_ThomasShehata_14MAR24_DIA_1_saliva1_sample"
    sample_name_mixture = ("[2] 2581_X_ThomasShehata_14MAR24_DIA_"
                           "14_saliva2_urine15_sample")
    assert not pure_is_in_mixture(sample_name_pure, sample_name_mixture)

    sample_name_pure = "[1] 2581_X_ThomasShehata_14MAR24_DIA_1_saliva1_sample"
    sample_name_mixture = ("[2] 2581_X_ThomasShehata_14MAR24_DIA_"
                           "14_saliva1_urine15.raw.PEP.Quantity")
    assert pure_is_in_mixture(sample_name_pure, sample_name_mixture)

