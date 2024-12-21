from pathlib import Path
from typing import Tuple, List

import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_memory
from umap import UMAP

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

        # Get sample columns
        sample_columns = get_sample_columns(df, indicator='SequencesUsedForQuantification')
        samples = df[sample_columns]
        samples = samples[samples > peptide_threshold].fillna(0).astype(bool)
        df[sample_columns] = samples

        # Drop nans
        df.dropna(subset=['PG.ProteinDescriptions'], inplace=True, axis=0)

        # Get abundance columns
        # abundance_columns = [c for c in df.columns if c.endswith('PG.Quantity')]
        # abundance_df = df[abundance_columns].where(np.array(samples))
        # df[sample_columns] = abundance_df
        # df.fillna(0, inplace=True)

        # Filter out proteins only in contaminant list
        df = df.loc[df['PG.Organisms'] != 'ContaminantList']

        # t-SNE variable picker
        grouping_var = st.selectbox("Grouping variable",
                                     options=['Temperature', 'Donor', 'Time'],
                                     key='tsne_grouping_variable')

        # Select sample columns
        temps = ['2', '21', '40']
        times = ['0h', '1day', '3days', '5days', '7days', '14days', '21days', '28days', '35days', '42days', '49days', '56days', '70days', '84days']
        donors = ['D1', 'D2', 'D3', 'D4', 'D5']
        checkboxes = {}

        grouping_vars = {"Temperature": temps,
                        "Time": times,
                        "Donor": donors}

        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("Temperatures to include")
            for t in temps:
                checkboxes[t] = st.checkbox(t, value=True)

        with c2:
            st.write("Times to include")
            for t in times:
                checkboxes[t] = st.checkbox(t, value=True)

        with c3:
            st.write("Donors to include")
            for d in donors:
                checkboxes[d] = st.checkbox(d, value=True)

        variables_to_keep = [c for (c, value) in zip(checkboxes.keys(), checkboxes.values()) if value]
        selected_sample_columns = []
        for c in sample_columns:
            from model import extract_variables
            vars = extract_variables(c)
            if (vars["Temperature"] in variables_to_keep and
                    vars["Donor"] in variables_to_keep and
                    vars["Time"] in variables_to_keep):
                    selected_sample_columns.append(c)

        # Prepare data
        x, y = prepare_data_new(df, indicator='SequencesUsedForQuantification', sample_columns=selected_sample_columns)

        # Run T-SNE
        x_embedded = TSNE(n_components=2, random_state=42, perplexity=min(len(selected_sample_columns)-2, 30)).fit_transform(x)

        # um = UMAP(n_components=2, n_neighbors=10, random_state=42)
        # x_embedded = um.fit_transform(x)

        # Store results in dataframe
        tsne_df = pd.DataFrame(x_embedded, columns=['x', 'y'])
        data = []
        for labels in y:
            data.append(labels[grouping_var])
        tsne_df['variable'] = data
        tsne_df['sample'] = selected_sample_columns

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

        # Calculate gini impurity and add to dataframe
        def group_columns_based_on_variable(df: pd.DataFrame,
                                            selected_columns: List[str],
                                            grouping_var: str) -> pd.DataFrame:

            # Create grouped columns
            grouped_columns = {}
            for t in grouping_vars[grouping_var]:
                grouped_columns[t] = []
                for c in selected_columns:
                    column_vars = extract_variables(c)
                    if column_vars[grouping_var] == t:
                        grouped_columns[t].append(c)

            # Overlay columns in the same group
            combined_columns = {}
            for group in grouped_columns.keys():
                combined_columns[group] = df[grouped_columns[group]].any(axis='columns')
            combined_columns['PG.ProteinDescriptions'] = df['PG.ProteinDescriptions']

            grouped_df = pd.DataFrame.from_dict(combined_columns)
            st.write(grouped_df)

            return grouped_df


        from analysis import gini_impurity
        df['gini impurity'] = group_columns_based_on_variable(df, selected_sample_columns, grouping_var).apply(
            lambda row: gini_impurity(np.array(row[grouping_vars[grouping_var]])), axis=1)

        st.write(df[['PG.ProteinDescriptions', 'gini impurity']].sort_values('gini impurity', ascending=True))

