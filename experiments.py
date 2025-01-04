import streamlit as st
from dotenv import load_dotenv
import phoenix_helpers
import helpers 


load_dotenv()
models = helpers.fetch_models()


if models:
    st.subheader("Select a Model")
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = models[0]
    st.selectbox(
        "Choose a model to use:", 
        models, key = 'selected_model',
        index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
    )

    datasetName = st.text_input("Dataset name")

    if datasetName:
        dataset = phoenix_helpers.get_dataset(name=datasetName)
        # df = dataset = dataset.as_dataframe()
        eval_res = phoenix_helpers.modelExperiment(st.session_state.selected_model,dataset)
        st.dataframe(eval_res)