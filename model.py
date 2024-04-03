import numpy as np
import sklearn.base
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Mapping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
import streamlit as st
from plotly.graph_objects import Figure
from skmultilearn.problem_transform import LabelPowerset

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

        # Get all proteins in the chosen samples and set random proteins to False
        fluid1_sample = protein_df[fluid1_sample]
        # fluid1_sample[fluid1_sample.sample(n=np.random.randint(0, len(fluid1_sample))).index] = False
        fluid2_sample = protein_df[fluid2_sample]
        # fluid2_sample[fluid2_sample.sample(n=np.random.randint(0, len(fluid2_sample))).index] = False

        # Combine new sample into single sample
        combined_sample = fluid1_sample | fluid2_sample

        # Define a name for the new sample
        sample_name = f'artificial{i}_{fluid1}_{fluid2}_sample'

        print(sample_name)

        # Add sample to dataframe as a new column
        protein_df[sample_name] = combined_sample

    return protein_df

def run_model(pure_protein_df: pd.DataFrame,
              mixed_protein_df: pd.DataFrame,
              model: str,
              n_artificial_samples: int = 0,
              identifying_proteins: pd.DataFrame = None):

    # Equalize dataframe dimensions
    mixed_protein_df = filter_on_train_proteins(pure_protein_df,
                                                mixed_protein_df)

    # Add simulated mixtures to train data
    pure_protein_df = add_simulated_mixtures(pure_protein_df,
                                             n=n_artificial_samples)

    # Prepare data
    x_train, y_train = prepare_data(pure_protein_df, multilabel=True)
    x_test, y_test = prepare_data(mixed_protein_df, multilabel=True)

    # Transform labels to label-indicator format
    mlb = MultiLabelBinarizer()
    y_train_transformed = mlb.fit_transform(y_train)
    y_test_transformed = mlb.transform(y_test)

    # Set correct estimator object
    if model == 'dt':
        estimator = DecisionTreeClassifier(random_state=42)
    elif model == 'rf':
        estimator = RandomForestClassifier(random_state=42)
    elif model == 'nn':
        estimator = MLPClassifier(random_state=42)

    # If identifying proteins used, fit separate model per fluid
    if identifying_proteins is not None:

        # Initialize model
        model = CustomOneVsRestClassifier(estimator,
                                          identifying_proteins)

        # Fit model
        model.fit(pure_protein_df)

        # Perform predictions
        predictions = model.predict(mixed_protein_df)

    # Otherwise use default sklearn OneVsRest model
    else:
        # Initialize model
        # model = OneVsRestClassifier(estimator)
        model = LabelPowerset(estimator)

        # Fit model
        model.fit(x_train, y_train_transformed)

        # Perform predictions
        predictions = model.predict(x_test)

    # Get metrics
    metrics = classification_report(y_test_transformed,
                                    predictions,
                                    output_dict=True,
                                    target_names=mlb.classes_)

    # Visualize metrics
    visualize_metrics(metrics)

    # Transform to dataframe
    # results_df = predictions_to_df(predictions_transformed, y_test)

    # Show confusion plot
    # TODO: Confusion plot

    # Tree graph
    # if isinstance(estimator, DecisionTreeClassifier):
    #     st.graphviz_chart(tree.export_graphviz(model.estimators_[0]))


class CustomOneVsRestClassifier:

    def __init__(self,
                 estimator,
                 identifying_proteins: pd.DataFrame):
        self._base_estimator = estimator
        self._identifying_proteins = identifying_proteins
        self._estimators = {}
        for fluid in BODY_FLUIDS:
            self._estimators[fluid] = sklearn.base.clone(estimator)

    def fit(self, protein_df):
        # Loop over fluids
        for fluid in BODY_FLUIDS:
            # Get identifying proteins for fluid (features)
            features = self._identifying_proteins.loc[
                (self._identifying_proteins['body fluid'] == fluid) & (self._identifying_proteins['% of samples with this protein'] > 80),
                'PG.ProteinDescriptions'
            ].to_list()

            # Prepare data
            x, y = prepare_data(protein_df, features)

            # Encode labels
            y = [1 if elem[0] == fluid else 0 for elem in y]

            # Store trained estimator
            self._estimators[fluid].fit(x, y)

    def predict(self, protein_df):
        # Get sample columns
        sample_columns = get_sample_columns(protein_df)

        # Store predictions
        predictions = np.empty(shape=(len(BODY_FLUIDS), len(sample_columns)))

        # Loop over fluids
        for i, fluid in enumerate(BODY_FLUIDS):
            # Get identifying proteins for fluid (features)
            features = self._identifying_proteins.loc[
                (self._identifying_proteins['body fluid'] == fluid) & (self._identifying_proteins['% of samples with this protein'] > 80),
                'PG.ProteinDescriptions'
            ].to_list()

            # Get trained estimator of fluid
            fluid_estimator = self._estimators[fluid]

            # Prepare data
            x, _ = prepare_data(protein_df, features)

            # Perform inference
            predictions[i] = fluid_estimator.predict(x)

        return predictions.T


class RelativeProteinFrequencyModel:

    def __init__(self):
        self.id_protein_counts = None

    def fit(self, identifying_proteins):
        # Get nr of identifying proteins per fluid
        self.id_protein_counts = identifying_proteins.groupby('body fluid').count()['PG.ProteinDescriptions']

    def predict(self, protein_df):
        # Get sample columns
        sample_columns = get_sample_columns(protein_df)

        # Store predictions
        predictions = []

        # Nr of proteins per sample
        sample_counts = protein_df[sample_columns].sum(axis=0)

        # Loop over samples
        for i in range(len(sample_counts)):
            # Divide nr of proteins in sample by nr of identifying proteins
            relative_nr = ((sample_counts[i] / self.id_protein_counts)
                           .sort_values(ascending=False))

            st.write(relative_nr)

            # Store
            predictions.append(relative_nr)

        return predictions
