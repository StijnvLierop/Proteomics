import pandas as pd
from typing import Iterable, Tuple
from matplotlib import pyplot as plt
import streamlit
from PIL import Image
import io
import plotly.express as px
from constants import BODY_FLUIDS


def columns_to_labels(column_names: Iterable[str]) -> Iterable[str]:
    labels = []
    for name in column_names:
        for fluid in BODY_FLUIDS:
            if fluid in name:
                labels.append(fluid)

    return labels


# https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image
def fig2img(fig: plt.Figure) -> Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def fig2data(fig: plt.Figure) -> io.BytesIO:
    output = io.BytesIO()
    fig2img(fig).save(output, format='PNG')
    return output


def column2fluid(column_name: str) -> list[str]:
    fluids = []
    for fluid in BODY_FLUIDS:
        if fluid in column_name:
            fluids.append(fluid)
    return fluids


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # Get sample columns
    sample_columns = [x for x in df.columns if x.endswith(".Quantity")]

    # Rename sample columns
    for column in sample_columns:
        df.rename(columns={column: column.replace('vagfluid', 'vaginalfluid')
                  .split(".")[0] + "_sample"},
                  inplace=True)

    return df

def preprocess_variations_df(df: pd.DataFrame) -> pd.DataFrame:
    # Get sample columns
    sample_columns = [x for x in df.columns if "PEP.Quantity" in x]

    # Rename sample columns
    for column in sample_columns:
        df.rename(columns={column: column.replace('vagfluid', 'vaginalfluid')
                  + "_sample"},
                  inplace=True)

    return df


def exclude_samples(df: pd.DataFrame,
                    samples_to_exclude: Iterable[str]) -> pd.DataFrame:
    # Remove samples to exclude from sample columns
    sample_columns = [x for x in df.columns if x not in samples_to_exclude]

    # Filter on remaining samples
    df = df[sample_columns]

    return df


def get_sample_columns(df: pd.DataFrame) -> list[str]:
    sample_columns = [x for x in df.columns if x.endswith("_sample")]
    return sample_columns


def style_df(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    float_columns = new_df.select_dtypes(include=['float']).columns
    new_df[float_columns] = new_df[float_columns].map('{:,.2f}'.format)
    return new_df


def pure_is_in_mixture(pure_sample: str, mix_sample: str) -> bool:
    # Make sure both sample names are non-empty
    if len(pure_sample) > 0 and len(mix_sample) > 0:
        # Extract identifier part from pure sample
        participant_id = pure_sample.split("_")[6]

        # Check if id is in mix sample name
        return participant_id in mix_sample

    # Return False by default
    return False


def get_unique_labels(body_fluids: Iterable[str]) -> list[str]:
    labels = []
    for fluid in body_fluids:
        labels.append(f"{fluid} in sample")
        labels.append(f"{fluid} predicted")

    return labels

def sample_name_to_participant_fluid(sample_name: str,
                                     fluid: str) -> str:
    elements = sample_name.split('_')
    result = [x for x in elements if fluid in x][0]
    return result

