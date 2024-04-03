import pandas as pd

from utils import preprocess_df
from analysis import filter_on_peptide_count
from model import filter_on_train_proteins, add_simulated_mixtures


def test_filter_on_train_proteins(pure_only_file, combi_only_file):
    po_pep = pd.read_excel(pure_only_file,
                           sheet_name='2581_PureOnly_Peptide')
    po_pep = preprocess_df(po_pep)
    pure_proteins = filter_on_peptide_count(po_pep, 3)

    co_pep = pd.read_excel(combi_only_file,
                           sheet_name='2581_CombiOnly_Peptide')
    co_pep = preprocess_df(co_pep)
    mix_proteins = filter_on_peptide_count(co_pep, 3)

    mix_proteins = filter_on_train_proteins(pure_proteins, mix_proteins)

    assert len(mix_proteins) == len(pure_proteins)
    assert (mix_proteins['PG.ProteinDescriptions'].to_list() ==
            pure_proteins['PG.ProteinDescriptions'].to_list())


def test_add_simulated_mixtures():
    pure_protein_df = pd.DataFrame(
        {'PG.ProteinDescriptions': ['prot1', 'prot2', 'prot3'],
         'saliva1_sample': [True, False, True],
         'blood1_sample': [False, False, True],
         'vaginalfluid_sample': [False, False, False],
         'semen_sample': [True, True, True],
         'urine_sample': [True, True, False]})

    pure_protein_df_aug = add_simulated_mixtures(protein_df=pure_protein_df,
                                                 n=1)

    print(pure_protein_df_aug)
