import pandas as pd
from typing import Iterable
from PIL import Image
import io

from constants import BODY_FLUIDS

def get_proteins_per_sample_true_in_mask(df: pd.DataFrame,
                                         mask: pd.DataFrame,
                                         fluid: str) -> pd.DataFrame:
    protein_df = pd.DataFrame(columns=['PG.ProteinDescriptions', 'body fluid'])
    for sample in mask.columns:
        proteins = df.loc[mask[sample], 'PG.ProteinDescriptions'].to_list()
        for protein in proteins:
            protein_df.loc[len(protein_df)] = [protein, fluid]
    return protein_df

def columns_to_labels(column_names: Iterable[str]) -> Iterable[str]:

    labels = []
    for name in column_names:
        for fluid in BODY_FLUIDS:
            if fluid in name:
                labels.append(fluid)

    return labels

# https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img