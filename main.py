from typing import Iterable
import numpy as np
import pandas as pd
import streamlit as st

# Define body fluid names
BODY_FLUIDS = ["saliva", "semen", "vaginalfluid", "urine", "blood"]

def add_protein_count_per_body_fluid(df: pd.DataFrame,
                                     samples_to_exclude: Iterable[str] = None) -> pd.DataFrame:
    # Define samples to look at
    samples = [x for x in df.columns if x.endswith("PG.Quantity")]
    if samples_to_exclude:
        samples = [x for x in samples if x not in samples_to_exclude]

    # For each protein and body fluid
    for key, protein_data in df.iterrows():
        for fluid in BODY_FLUIDS:
            # Initialize count to 0
            count = 0

            # Loop over samples
            for sample in samples:
                # If protein intensity is not NaN, protein is present in sample so add 1 to the count for the current fluid
                if fluid in sample and not np.isnan(float(str(protein_data[sample]).replace(',', '.'))):
                    count += 1

            # Store count for current protein and fluid in dataframe
            df.loc[key, fluid] = count

    return df


if __name__ == '__main__':
    # Set samples to exclude
    samples_to_exclude = None

    # Upload file
    st.header("Upload bestanden")
    po_file = st.file_uploader(label="PureOnly bestand", type='.xlsx')

    # When file uploaded
    if po_file is not None:

        # Read into dataframe
        po_df = pd.read_excel(po_file)

        # Add nr of times each protein occurs in a body fluid
        po_df = add_protein_count_per_body_fluid(po_df, samples_to_exclude)

        # Show dataframe
        show_df = po_df[['PG.ProteinDescriptions'] + BODY_FLUIDS]
        st.write("Nr of samples with protein per body fluid")
        st.write(show_df)