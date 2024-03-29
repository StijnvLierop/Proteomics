import streamlit as st
import pandas as pd

from analysis import filter_on_peptide_count, \
    get_protein_count_per_body_fluid, \
    add_gini_impurity, \
    get_identifying_proteins_per_body_fluid, \
    get_protein_differences_pure_sample_with_mixture
from constants import BODY_FLUIDS
from visualize import run_TSNE_pure


if __name__ == '__main__':
    # Set wide page layout
    st.set_page_config(layout="wide")

    # Set samples to exclude
    samples_to_exclude = None

    # Upload file
    st.header("Upload files")
    pure_file = st.file_uploader(label="PureOnly file",
                                 type='.xlsx')
    combi_file = st.file_uploader(label="CombiOnly file",
                                  type='.xlsx')

    # Peptide threshold
    peptide_threshold = st.number_input("Peptide threshold (nr of peptides >= "
                                        "peptide threshold)", value=3)

    # When file uploaded
    if pure_file is not None and combi_file is not None:
        # Only read files again if they are not yet in memory
        if ('pure_peptide_df' not in st.session_state or
                'mixture_peptide_df' not in st.session_state):
            # Read into dataframe
            st.session_state['pure_peptide_df'] = (
                pd.read_excel(pure_file, sheet_name='2581_PureOnly_Peptide'))
            st.session_state['mixture_peptide_df'] = (
                pd.read_excel(combi_file, sheet_name='2581_CombiOnly_Peptide'))

        #
        st.session_state["sample_columns"] = [x for x in st.session_state['pure_peptide_df'].columns
                                              if x.endswith("PEP.Quantity")]

        # Filter on proteins that have at least n detected peptides per sample
        if 'proteins_per_pure_sample' not in st.session_state:
            st.session_state['proteins_per_pure_sample'] = (
                filter_on_peptide_count(st.session_state['pure_peptide_df'],
                                        peptide_threshold))
        if 'proteins_per_mixture_sample' not in st.session_state:
            st.session_state['proteins_per_mixture_sample'] = (
                filter_on_peptide_count(st.session_state['mixture_peptide_df'],
                                        peptide_threshold))

        # Add nr of times each protein occurs in a body fluid
        if 'nr_of_samples_with_protein_per_body_fluid' not in st.session_state:
            st.session_state['nr_of_samples_with_protein_per_body_fluid'] = (
                get_protein_count_per_body_fluid(
                    st.session_state['proteins_per_pure_sample'],
                                        samples_to_exclude)
            )

        # Calculate information gain per protein
        if ('nr_of_samples_with_protein_per_body_fluid' not in st.session_state
                or 'proteins_in_no_samples_per_body_fluid' not in st.session_state):
            (st.session_state['nr_of_samples_with_protein_per_body_fluid'],
             st.session_state['proteins_in_no_samples_per_body_fluid']) = add_gini_impurity(
                st.session_state['nr_of_samples_with_protein_per_body_fluid'])

        # Show nr of samples with protein per body fluid
        st.write("Relative nr of samples with protein per body fluid")
        st.write(st.session_state['nr_of_samples_with_protein_per_body_fluid'])

        # Get identifying proteins per body fluid
        if 'identifying_proteins' not in st.session_state:
            st.session_state['identifying_proteins'] = (
                get_identifying_proteins_per_body_fluid(
                st.session_state['nr_of_samples_with_protein_per_body_fluid'])
            )

        # Get differences in proteins between pure sample and mixture
        if ('proteins_in_mixture_not_in_pure_sample' not in st.session_state or
                'proteins_in_pure_sample_not_in_mixture' not in st.session_state):
            (st.session_state['proteins_in_mixture_not_in_pure_sample'],
             st.session_state['proteins_in_pure_sample_not_in_mixture']) = (
                get_protein_differences_pure_sample_with_mixture(
                    st.session_state['proteins_per_pure_sample'],
                    st.session_state['proteins_per_mixture_sample']))

        # Show specific results per fluid
        st.header("Analysis per body fluid")

        # Fluid selection
        selected_fluid = st.radio("Body fluid selection",
                                  BODY_FLUIDS,
                                  horizontal=True)
        if selected_fluid is not None:
            # Show proteins specific for certain body fluids
            with st.expander("Identifying proteins", expanded=True):
                st.write(st.session_state['identifying_proteins']
                         [st.session_state['identifying_proteins']
                          ['body fluid'] == selected_fluid]
                         .drop('body fluid', axis=1)
                         )

            # Show proteins that are present in mixture sample,
            # but not in pure sample
            with st.expander(
                    f"Proteins in mixture not in pure sample for {selected_fluid}",
                    expanded=True):
                st.table(
                    st.session_state['proteins_in_mixture_not_in_pure_sample']
                    [st.session_state['proteins_in_mixture_not_in_pure_sample']
                     ['body fluid'] == selected_fluid]
                    .drop('body fluid', axis=1)
                )

            # Show proteins that are present in pure sample,
            # but not in mixture sample
            with st.expander(
                    f"Proteins in pure sample not in mixture for {selected_fluid}",
                    expanded=True):
                st.table(
                    st.session_state['proteins_in_pure_sample_not_in_mixture']
                    [st.session_state['proteins_in_pure_sample_not_in_mixture']
                     ['body fluid'] == selected_fluid]
                    .drop('body fluid', axis=1)
                )

        # Show T-SNE plot of pure samples
        st.image(run_TSNE_pure(st.session_state["proteins_per_pure_sample"], st.session_state["sample_columns"]))
