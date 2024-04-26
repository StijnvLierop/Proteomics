import pandas as pd
import streamlit as st

from analysis import filter_on_chemical_vars, get_sample_agreements
from utils import (get_sample_columns,
                   preprocess_variations_df,
                   sample_name_to_participant_fluid)
from sklearn.metrics import accuracy_score, top_k_accuracy_score

if __name__ == '__main__':
    # Set wide page layout
    st.set_page_config(layout="wide")

    # Upload file
    st.header("Upload file")
    var_file = st.file_uploader(label="Variations file",
                                type='.xlsx')

    # Only continue when file has been uploaded
    if var_file is not None:
        # Read data
        df = pd.read_excel(var_file)

        # Preprocess df
        df = preprocess_variations_df(df)

        # Get sample columns
        sample_columns = get_sample_columns(df)

        # Binarize samples
        df[sample_columns] = df[sample_columns].fillna(0)
        df[df[sample_columns] > 0] = 1

        # Get operations
        df['Variation'] = df.apply(lambda x: str(x['PG.ProteinAccessions'])
                                   .split('.')[-1], axis=1)

        # Filter on chemical variations
        df = filter_on_chemical_vars(df)

        # Get variations per participant
        sample_agreements = get_sample_agreements(df)

        # Transform sample agreements to classifications
        predictions = [sample_name_to_participant_fluid(x, 'saliva')
                       for x in sample_agreements.idxmax(axis=1).to_list()]

        # True values
        true_values = [sample_name_to_participant_fluid(x, 'saliva') for x
                       in sample_agreements.index.to_list()]

        st.write("Proportion of variations in common between "
                 "mix and each pure sample:")
        st.write(sample_agreements)

        # Print metrics
        st.write("When using highest agreement score as prediction:")
        st.write("Accuracy:", accuracy_score(true_values, predictions))
