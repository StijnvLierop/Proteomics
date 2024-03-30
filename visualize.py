import collections

import pandas as pd
from typing import Iterable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure

from utils import columns_to_labels, fig2img, get_sample_columns, column2fluid


def run_tsne_pure(proteins_per_pure_sample: pd.DataFrame) -> Figure:
    # Get sample columns
    sample_columns = get_sample_columns(proteins_per_pure_sample)

    # Create feature matrix n_samples x n_proteins
    x = proteins_per_pure_sample[sample_columns].T

    # Create label vector of body fluids
    y = columns_to_labels(proteins_per_pure_sample[sample_columns].columns)

    # Run T-SNE
    x_embedded = TSNE(n_components=2, random_state=42).fit_transform(x)

    # Store results in dataframe
    df = pd.DataFrame(x_embedded, columns=['x', 'y'])
    df['body fluid'] = y

    # Visualize results
    fig = px.scatter(df,
                     x='x',
                     y='y',
                     color='body fluid',
                     title="T-SNE projection of pure samples")

    return fig


def protein_counts_per_fluid_dist(proteins_per_pure_sample: pd.DataFrame) \
        -> Figure:
    # Get sample columns
    sample_columns = get_sample_columns(proteins_per_pure_sample)
    fluids = [column2fluid(x) for x in sample_columns]
    fluid_counts = collections.Counter(fluids)

    # Add n to fluids
    fluids = [f"{fluid} (n={fluid_counts[fluid]})" for fluid in fluids]

    # Get information on nr of different proteins found per body fluid
    protein_counts = proteins_per_pure_sample[sample_columns].sum(axis=0)
    protein_counts_per_fluid = pd.DataFrame(
        np.array([protein_counts.values, fluids]).T,
        columns=['count', 'fluid'])

    # Make figure
    fig = px.box(protein_counts_per_fluid,
                 x='count',
                 color='fluid',
                 title="Nr of proteins found",
                 labels={'count': "Nr of proteins",
                         'fluid': "Body fluid"})

    return fig
