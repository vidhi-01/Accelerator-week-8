import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("🤖 LLM Evaluation")
pages = [
        st.Page("GenerateModelTraces.py", title="Generate Traces"),
        st.Page("LLMasjudge.py", title="LLM as a Judge"),
        st.Page("evaluationwithexistingdata.py", title="Evaluate with existing data"),
        st.Page("experiments.py", title="Run Experiment"),
    ]

pg = st.navigation(pages)
pg.run()


# action = st.radio("What would you like to do?", ["Generate Traces for Model", "Evaluate Model"])
# models = helpers.fetch_models()
# if models:
#     if action == "Generate Traces for Model":
#         st.subheader("Select a Model")
#         if "selected_model" not in st.session_state:
#             st.session_state.selected_model = models[0]
#         st.selectbox(
#             "Choose a model to use:", 
#             models, key = 'selected_model',
#             index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
#         )

#         if st.session_state.selected_model:
#             st.subheader("Enter a Prompt")
#             if "prompt" not in st.session_state:
#                 st.session_state.prompt = ""
#             st.session_state.prompt = st.text_area("Enter your prompt:", value=st.session_state.prompt)
#             if st.button("Generate Content", on_click=callback):
#                 if st.session_state.prompt:
#                     st.subheader("Model Output")
#                     st.session_state.generated_content = helpers.generate_content(st.session_state.selected_model, st.session_state.prompt)
#                     st.write(st.session_state.generated_content)
#                     st.session_state.spans_df = phoenix_helpers.get_spans_df()
#                     # print(spans_df)
#                     st.dataframe(st.session_state.spans_df)
#                 else:
#                     st.write("Enter something to generate content.")
#     elif action == "Evaluate Model":
#         st.session_state.spans_df = phoenix_helpers.get_spans_df()
#         st.subheader("Evaluate LLM")

#         if "evaluation_result" not in st.session_state:
#             st.session_state.evaluation_result = None
#         if (st.button("Evaluate", on_click=callback2) or st.session_state.eval_btn_clicked):
#             if "eval_model" not in st.session_state:
#                 st.session_state.eval_model = models[0]
#             st.selectbox(
#                 "Choose a model to use for evaluation:", 
#                 models, key = 'eval_model',
#                 index=models.index(st.session_state.eval_model) if st.session_state.eval_model in models else 0,
#             )
#             if st.session_state.eval_model:
#                 st.session_state.evaluation_result = phoenix_helpers.evaluate_model(st.session_state.spans_df, st.session_state.eval_model)
#                 st.write(st.session_state.evaluation_result)

                    
