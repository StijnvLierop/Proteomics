import numpy as np
import sklearn.base
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Tuple, Mapping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
import streamlit as st
from plotly.graph_objects import Figure
from skmultilearn.problem_transform import LabelPowerset
from sklearn.feature_selection import VarianceThreshold
from utils import get_sample_columns, column2fluid, get_unique_labels
from constants import BODY_FLUIDS
from visualize import visualize_metrics, visualize_tsne


def run_tsne(pure_proteins: pd.DataFrame,
             mix_proteins: pd.DataFrame) -> None:
    # Get common proteins
    mix_proteins = filter_on_train_proteins(pure_proteins, mix_proteins)

    # Get data
    x_pure, y_pure = prepare_data(pure_proteins, multilabel=True)
    x_mix, y_mix = prepare_data(mix_proteins, multilabel=True)
    x = np.concatenate((x_pure, x_mix), axis=0)
    y_pure.extend(y_mix)
    y = y_pure

    # Run T-SNE
    x_embedded = TSNE(n_components=2, random_state=42).fit_transform(x)

    # Convert labels
    mlb = MultiLabelBinarizer()
    mlb.fit(y_pure)
    y_transformed = mlb.transform(y)

    # Store results in dataframe
    df = pd.DataFrame(x_embedded, columns=['x', 'y'])
    fluid_data = []
    for labels in y:
        if len(labels) > 1:
            fluid_data.append(', '.join(labels))
        else:
            fluid_data.append(labels[0])
    df['fluid'] = fluid_data

    # Visualize
    visualize_tsne(df)

def prepare_data(protein_df: pd.DataFrame,
                 filter_proteins: list[str] = None,
                 multilabel: bool = True) -> Tuple[np.ndarray, list[str]]:
    # Get sample columns
    sample_columns = get_sample_columns(protein_df)

    # If identifying proteins, filter rows on these proteins
    if filter_proteins:
        protein_df = protein_df[
            protein_df['PG.ProteinDescriptions'].isin(filter_proteins)
        ]

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

    # Fill nans with False
    mix_protein_df = mix_protein_df.fillna(False)

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


def add_simulated_mixtures(protein_df: pd.DataFrame,
                           n: int) -> pd.DataFrame:
    # Get sample columns
    sample_columns = get_sample_columns(protein_df)

    # Loop over nr of artificial mixtures to add
    for i in range(n):
        # Choose 2 random fluids
        fluid1, fluid2 = np.random.choice(BODY_FLUIDS, 2, replace=False)
        print(fluid1, fluid2)

        # Choose a random other sample per fluid
        fluid1_sample = np.random.choice([x for x in sample_columns
                                          if fluid1 in x],
                                         1)[0]
        fluid2_sample = np.random.choice([x for x in sample_columns
                                          if fluid2 in x],
                                         1)[0]

        # Get all proteins in the chosen samples
        # and set random proteins to False
        fluid1_sample = protein_df[fluid1_sample]
        # fluid1_sample[fluid1_sample.sample(n=np.random.randint(0, len(fluid1_sample))).index] = False
        fluid2_sample = protein_df[fluid2_sample]
        # fluid2_sample[fluid2_sample.sample(n=np.random.randint(0, len(fluid2_sample))).index] = False

        # Combine new sample into single sample
        combined_sample = fluid1_sample | fluid2_sample

        # Define a name for the new sample
        sample_name = f'artificial{i}_{fluid1}_{fluid2}_sample'

        # Add sample to dataframe as a new column
        protein_df[sample_name] = combined_sample

    return protein_df

class RelativeProteinFrequencyModel:
    def fit(self, identifying_proteins):
        # Get nr of identifying proteins per fluid
        self.identifying_proteins = identifying_proteins

    def predict(self, protein_df):
        # Get sample columns
        sample_columns = get_sample_columns(protein_df)

        # Store predictions
        predictions = []

        # Loop over samples
        for sample in sample_columns:
            relative_proteins = {}
            for fluid in BODY_FLUIDS:
                id_proteins = self.identifying_proteins.loc[self.identifying_proteins['body fluid'] == fluid, 'PG.ProteinDescriptions'].to_list()
                mix_proteins = protein_df.loc[protein_df[sample], 'PG.ProteinDescriptions'].to_list()
                overlap_proteins = set(id_proteins).intersection(set(mix_proteins))
                relative_proteins[fluid] = len(overlap_proteins) / len(id_proteins)

            # Select top 2 as present in fluid
            selected_fluids = [x[0] for x in sorted(relative_proteins.items(), key=lambda x:x[1])[-2:]]

            predictions.append(selected_fluids)

        return predictions
