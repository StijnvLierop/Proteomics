import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st

from utils import get_sample_columns, column2fluid, get_unique_labels
from constants import BODY_FLUIDS


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


def filter_on_train_proteins(pure_protein_df: pd.DataFrame,
                             mix_protein_df: pd.DataFrame) \
        -> pd.DataFrame:
    # Mix df sample columns
    mix_sample_columns = get_sample_columns(mix_protein_df)

    # Get proteins to add
    df_all = pure_protein_df.merge(mix_protein_df.drop_duplicates(),
                                   on=['PG.ProteinDescriptions',
                                       'PG.Genes',
                                       'PG.ProteinAccessions'],
                                   how='left',
                                   indicator=True)
    proteins_to_add = df_all.loc[
        df_all['_merge'] == 'left_only',
        ['PG.ProteinDescriptions', 'PG.Genes', 'PG.ProteinAccessions'] +
        mix_sample_columns
    ]

    # Keep proteins in mix also in pure
    mix_protein_df = mix_protein_df.merge(pure_protein_df.drop_duplicates(),
                                          on=['PG.ProteinDescriptions',
                                              'PG.Genes',
                                              'PG.ProteinAccessions'],
                                          how='inner')
    mix_protein_df = mix_protein_df[['PG.ProteinDescriptions',
                                     'PG.Genes',
                                     'PG.ProteinAccessions']
                                    + mix_sample_columns]

    # Add proteins
    mix_protein_df = pd.concat([mix_protein_df, proteins_to_add])

    return mix_protein_df


def predictions_to_df(predictions: list[str], y_true: list[str]) \
        -> pd.DataFrame:
    # Initialize dataframe to store results
    labels = get_unique_labels(BODY_FLUIDS)
    data = np.zeros(shape=(len(predictions), len(labels))).astype(bool)
    results_df = pd.DataFrame(data, columns=labels)
    index = results_df.index

    # Loop over samples
    for index, sample_pred, sample_true in zip(index, predictions, y_true):
        # Set predicted fluids in sample to true
        for pred_fluid in sample_pred:
            results_df.loc[index, f"{pred_fluid} predicted"] = True

        # Set true fluids in sample to true
        for true_fluid in sample_true:
            results_df.loc[index, f"{true_fluid} in sample"] = True

    return results_df


def run_decision_tree(pure_protein_df: pd.DataFrame,
                      mixed_protein_df: pd.DataFrame) -> str:
    # Equalize dataframe dimensions
    mixed_protein_df = filter_on_train_proteins(pure_protein_df,
                                                mixed_protein_df)

    # Prepare data
    x_train, y_train = prepare_data(pure_protein_df, multilabel=True)
    x_test, y_test = prepare_data(mixed_protein_df, multilabel=True)

    # Transform labels to label-indicator format
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)

    # Initialize model
    model = DecisionTreeClassifier(random_state=42)

    # Fit model
    model.fit(x_train, y_train)

    # Perform predictions
    predictions = model.predict(x_test)

    # Get label predictions
    predictions = mlb.inverse_transform(predictions)

    # Transform to dataframe
    results_df = predictions_to_df(predictions, y_test)
    st.write(results_df)

    # Tree graph
    graph = tree.export_graphviz(model)

    return graph
