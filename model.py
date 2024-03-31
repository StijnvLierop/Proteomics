import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st

from utils import get_sample_columns, column2fluid


def prepare_data(protein_df: pd.DataFrame,
                 multilabel: bool = True) -> Tuple[np.ndarray, list[str]]:
    # Get sample columns
    sample_columns = get_sample_columns(protein_df)

    # Define feature vector x
    x = np.array(protein_df[sample_columns].T)

    # Get labels
    if multilabel:
        y = [column2fluid(x) for x in sample_columns]
    else:
        y = [column2fluid(x)[0] for x in sample_columns]

    return x, y


def equalize_dimensions(pure_protein_df: pd.DataFrame,
                        mix_protein_df: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Get sample columns
    pure_sample_columns = get_sample_columns(pure_protein_df)
    mix_sample_columns = get_sample_columns(mix_protein_df)

    # Get proteins in both dataframes
    pure_proteins = set(pure_protein_df['PG.ProteinDescriptions'].to_list())
    mix_proteins = set(mix_protein_df['PG.ProteinDescriptions'].to_list())

    # Get proteins that have to be added to both dataframes
    add_to_pure = list(mix_proteins - pure_proteins)
    add_to_mix = list(pure_proteins - mix_proteins)

    # Add required proteins to both dataframes, set sample columns to False
    if len(add_to_pure) > 0:
        add_to_pure = mix_protein_df[
            mix_protein_df['PG.ProteinDescriptions'].isin(add_to_pure)]
        add_to_pure[mix_sample_columns] = False
        pure_protein_df = pd.concat([pure_protein_df, add_to_pure])

    if len(add_to_mix) > 0:
        add_to_mix = pure_protein_df[
            pure_protein_df['PG.ProteinDescriptions'].isin(add_to_mix)]
        add_to_mix[pure_sample_columns] = False
        mix_protein_df = pd.concat([mix_protein_df, add_to_mix])

    # st.write(pure_protein_df)
    # st.write(mix_protein_df)

    return pure_protein_df, mix_protein_df


def run_decision_tree(pure_protein_df: pd.DataFrame,
                      mixed_protein_df: pd.DataFrame) -> str:
    # Equalize dataframe dimensions
    pure_protein_df, mixed_protein_df = equalize_dimensions(pure_protein_df,
                                                            mixed_protein_df)

    # Prepare data
    x_train, y_train = prepare_data(pure_protein_df, multilabel=True)
    x_test, y_test = prepare_data(mixed_protein_df, multilabel=True)

    # Transform labels to label-indicator format
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)

    # Initialize model
    model = DecisionTreeClassifier(random_state=42)

    # Fit model
    model.fit(x_train, y_train)

    # Perform predictions
    predictions = model.predict(x_test)

    st.write(y_test, predictions)

    # Tree graph
    graph = tree.export_graphviz(model)

    return graph
