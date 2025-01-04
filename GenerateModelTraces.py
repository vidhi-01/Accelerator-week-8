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

    if st.session_state.selected_model:
        st.subheader("Enter a Prompt")
        if "prompt" not in st.session_state:
            st.session_state.prompt = ""
        st.session_state.prompt = st.text_area("Enter your prompt:", value=st.session_state.prompt)
        if st.button("Generate Content"):
            if st.session_state.prompt:
                st.subheader("Model Output")
                st.session_state.generated_content = helpers.generate_content(st.session_state.selected_model, st.session_state.prompt)
                st.write(st.session_state.generated_content)
                st.session_state.spans_df = phoenix_helpers.get_spans_df()
                # print(spans_df)
                st.dataframe(st.session_state.spans_df)
                st.session_state.prompt = ""
            else:
                st.write("Enter something to generate content.")


                
