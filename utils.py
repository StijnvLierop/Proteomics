import pandas as pd


def get_proteins_per_sample_true_in_mask(df: pd.DataFrame,
                                         mask: pd.DataFrame,
                                         fluid: str) -> pd.DataFrame:
    protein_df = pd.DataFrame(columns=['PG.ProteinDescriptions', 'body fluid'])
    for sample in mask.columns:
        proteins = df.loc[mask[sample], 'PG.ProteinDescriptions'].to_list()
        for protein in proteins:
            protein_df.loc[len(protein_df)] = [protein, fluid]
    return protein_df
