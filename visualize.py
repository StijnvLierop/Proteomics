import collections

import seaborn
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

from utils import columns_to_labels, fig2data, get_sample_columns, column2fluid
from constants import BODY_FLUIDS, COLOR_MAPPING

# Set font Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Set output quality
plt.rcParams['figure.dpi'] = 600


def visualize_tsne(df: pd.DataFrame) -> None:
    # Visualize results
    interactive_fig = px.scatter(df,
                                 x='x',
                                 y='y',
                                 color='body fluid',
                                 title="t-SNE projection of pure samples",
                                 labels={'x': 't-SNE feature 1',
                                         'y': 't-SNE feature 2'},
                                 color_discrete_map=COLOR_MAPPING,
                                 template='plotly')
    interactive_fig.update_layout(title_x=0.25)
    interactive_fig.update_layout(font_family='Times New Roman',
                                  title_font_family='Times New Roman',
                                  legend_title_text='Body fluid')
    interactive_fig.update_xaxes(showgrid=True)
    interactive_fig.update_yaxes(showgrid=True)
    st.plotly_chart(interactive_fig)

    import io
    # # image_data = io.BytesIO()
    # image_data = interactive_fig.to_image(format='PNG')
    #
    # st.download_button("Download Figure",
    #                    data=image_data,
    #                    mime="image/png",
    #                    key="t-SNE_pure_figure")

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
        -> None:
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
    interactive_fig = px.box(protein_counts_per_fluid,
                             x='count',
                             color='fluid',
                             title="Number of identified proteins "
                                   "per body fluid",
                             labels={'count': "Number of proteins",
                                     'fluid': "Body fluid"},
                             color_discrete_map=COLOR_MAPPING)
    interactive_fig.update_layout(title_x=0.25)
    interactive_fig.update_layout(font_family='Times New Roman',
                                  title_font_family='Times New Roman')
    interactive_fig.update_xaxes(showgrid=True)
    interactive_fig.update_yaxes(showgrid=False)
    st.plotly_chart(interactive_fig)

    # # Make publication figure
    # pub_figure, ax = plt.subplots(figsize=(8, 4))
    # sns.boxplot(protein_counts_per_fluid,
    #             x='count',
    #             hue='fluid',
    #             ax=ax,
    #             palette=COLOR_MAPPING,
    #             zorder=3)
    # sns.despine(pub_figure)
    # ax.set(xlabel='Number of proteins',
    #        title="Number of identified proteins per body fluid")
    # ax.grid(axis='x', zorder=0)
    # plt.legend(title='Body fluid',
    #            bbox_to_anchor=(1.3, 0.5),
    #            loc='center right')
    # plt.tight_layout()
    #
    # # Enable download button
    # figure_data = fig2data(pub_figure)
    # st.download_button("Download Figure",
    #                    data=figure_data,
    #                    mime="image/png",
    #                    key="protein_count_figure")


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
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig)

    # Plot Support
    with col2:
        fig = px.bar(df.loc[df.index.isin(BODY_FLUIDS), 'support'],
                     labels={'index': 'Body Fluid', 'value': 'Support'},
                     title='Support per class')
        fig.update_layout(showlegend=False)
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig)

    # Plot Precision
    with col3:
        fig = px.bar(df['precision'],
                     labels={'index': 'Body Fluid', 'value': 'Precision'},
                     title='Precision per class')
        fig.update_layout(showlegend=False)
        fig.update_layout(yaxis_range=[0, 1])
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig)

    # Plot Recall
    with col4:
        fig = px.bar(df['recall'],
                     labels={'index': 'Body Fluid', 'value': 'Recall'},
                     title='Recall per class')
        fig.update_layout(showlegend=False)
        fig.update_layout(yaxis_range=[0, 1])
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig)
