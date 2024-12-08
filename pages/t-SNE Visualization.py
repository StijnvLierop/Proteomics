from pathlib import Path
from typing import Tuple

import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer

from model import prepare_data_new
from analysis import filter_on_peptide_count
from utils import get_sample_columns
from visualize import visualize_tsne

@st.cache_data
def load_data(pure_file_path: Path) -> pd.DataFrame:
    pure_peptide_df = pd.read_excel(pure_file_path,
                                    sheet_name='proteins_1')

    return pure_peptide_df


if __name__ == '__main__':
    # Set wide page layout
    st.set_page_config(layout="wide")

    # Set samples to exclude
    st.session_state['samples_to_exclude'] = None

    # Upload file
    st.header("Upload files")
    pure_file = st.file_uploader(label="PureOnly file",
                                 type='.xlsx')

    # Peptide threshold
    peptide_threshold = st.number_input("Peptide threshold "
                                        "(nr of peptides >= "
                                        "peptide threshold)",
                                        value=3,
                                        key='peptide_threshold')

    if pure_file:
        df = load_data(pure_file)

        sample_columns = get_sample_columns(df, indicator='SequencesUsedForQuantification')
        samples = df[sample_columns]
        samples = samples[samples > peptide_threshold].fillna(0).astype(bool)
        df[sample_columns] = samples

        # t-SNE variable picker
        grouping_var = st.selectbox("Grouping variable",
                                         options=['Temperature', 'Donor', 'Time','Unknown'],
                                         key='tsne_grouping_variable')

        # Prepare data
        x, y = prepare_data_new(df, indicator='SequencesUsedForQuantification')

        # Run T-SNE
        x_embedded = TSNE(n_components=2, random_state=42).fit_transform(x)

        # Store results in dataframe
        tsne_df = pd.DataFrame(x_embedded, columns=['x', 'y'])
        data = []
        for labels in y:
            data.append(labels[grouping_var])
        tsne_df['variable'] = data
        tsne_df['sample'] = get_sample_columns(df, indicator='SequencesUsedForQuantification')

        # Visualize
        interactive_fig = px.scatter(tsne_df,
                                     x='x',
                                     y='y',
                                     color='variable',
                                     title="t-SNE projection of pure samples",
                                     labels={'x': 't-SNE feature 1',
                                             'y': 't-SNE feature 2'},
                                     hover_data=["sample"],
                                     template='plotly')
        interactive_fig.update_layout(title_x=0.25)
        interactive_fig.update_layout(font_family='Times New Roman',
                                      title_font_family='Times New Roman',
                                      legend_title_text='variable')
        interactive_fig.update_xaxes(showgrid=True)
        interactive_fig.update_yaxes(showgrid=True)
        st.plotly_chart(interactive_fig)



