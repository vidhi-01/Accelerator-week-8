import streamlit as st
from dotenv import load_dotenv
import phoenix_helpers
import helpers 
import plotly.express as px


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
        df = dataset = dataset.as_dataframe()
        eval_res = phoenix_helpers.dataEvalResults(st.session_state.selected_model,df)

        eval_res["hallucination_eval"] = eval_res["hallucination_eval"].apply(
            lambda x: x if x in ["hallucinated", "factual"] else "other"
        )
        eval_res["qa_eval"] = eval_res["qa_eval"].apply(
            lambda x: x if x in ["correct", "incorrect"] else "other"
        )

        response_counts = eval_res["hallucination_eval"].value_counts().reset_index()
        response_counts.columns = ["hallucination_eval", "count"]
        response_counts_qa = eval_res["qa_eval"].value_counts().reset_index()
        response_counts_qa.columns = ["qa_eval", "count"]

        st.dataframe(eval_res)

        fig = px.pie(
            response_counts,
            values="count",
            names="hallucination_eval",
            title=f"{st.session_state.selected_model} Model Hallucination",
            hole=0.4  
        )

        st.plotly_chart(fig)
        
        fig = px.pie(
            response_counts_qa,
            values="count",
            names="qa_eval",
            title=f"{st.session_state.selected_model} Model QA",
            hole=0.4  
        )

        st.plotly_chart(fig)

    
    