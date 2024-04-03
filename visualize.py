import collections
import streamlit as st
import pandas as pd
from typing import Iterable
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure
from typing import Tuple, Mapping

from utils import columns_to_labels, fig2img, get_sample_columns, column2fluid
from constants import BODY_FLUIDS


def visualize_tsne(df: pd.DataFrame) -> Figure:
    print(df)

    # Visualize results
    fig = px.scatter(df,
                     x='x',
                     y='y',
                     color='fluid',
                     title="T-SNE projection of pure samples")

    st.plotly_chart(fig)


def protein_counts_per_fluid_dist(proteins_per_pure_sample: pd.DataFrame) \
        -> Figure:
    # Get sample columns
    sample_columns = get_sample_columns(proteins_per_pure_sample)
    fluids = [column2fluid(x)[0] for x in sample_columns]
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


def visualize_metrics(metrics: dict[str, float]) -> None:
    # Create df for plotting
    df = pd.DataFrame.from_dict(metrics,
                                orient='index')

    # Give header
    st.subheader("Metrics")

    # Show in three columns
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Plot F1-Score
    with col1:
        fig = px.bar(df['f1-score'],
                     labels={'index': 'Body Fluid', 'value': 'F1-Score'},
                     title='F1-Score per class')
        fig.update_layout(showlegend=False)
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig)

    # Plot Support
    with col2:
        fig = px.bar(df.loc[df.index.isin(BODY_FLUIDS), 'support'],
                     labels={'index': 'Body Fluid', 'value': 'Support'},
                     title='Support per class')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

    # Plot Precision
    with col3:
        fig = px.bar(df['precision'],
                     labels={'index': 'Body Fluid', 'value': 'Precision'},
                     title='Precision per class')
        fig.update_layout(showlegend=False)
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig)

    # Plot Recall
    with col4:
        fig = px.bar(df['recall'],
                     labels={'index': 'Body Fluid', 'value': 'Recall'},
                     title='Recall per class')
        fig.update_layout(showlegend=False)
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig)
