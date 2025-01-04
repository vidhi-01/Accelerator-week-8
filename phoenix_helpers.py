from phoenix.otel import register
from openinference.instrumentation.groq import GroqInstrumentor
import phoenix as px
import pandas as pd
from dotenv import load_dotenv
from phoenix.trace import SpanEvaluations
from phoenix.evals import HallucinationEvaluator, LiteLLMModel, QAEvaluator, run_evals
import json
from phoenix.trace.dsl import SpanQuery
load_dotenv()
import os
from sentence_transformers import SentenceTransformer, util
from phoenix.experiments import run_experiment

from groq import Groq

SentenceTransformer_model = SentenceTransformer('all-MiniLM-L6-v2')

client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)


tracer_provider = register(
  project_name="default",
  endpoint="https://app.phoenix.arize.com/v1/traces",
) 

GroqInstrumentor().instrument(tracer_provider=tracer_provider)

def process_output_json_column(json_str):
    try:
        # Parse the JSON string
        parsed = json.loads(json_str)
        # Extract desired fields
        total_time = parsed.get("usage", {}).get("total_time", None)
        total_tokens = parsed.get("usage", {}).get("total_tokens", None)
        model = parsed.get("model", None)
        message_content = (
            parsed.get("choices", [{}])[0]
            .get("message", {})
            .get("content", None)
        )
        return total_time, total_tokens, model, message_content
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, None, None, None
def process_input_json_column(json_str):
    try:
        parsed = json.loads(json_str)
        message_content = parsed.get("messages", {})[0].get("content", None)
        
        return message_content
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
def process_json_column_with_data(json_str):
    # print("Asdasd",json_str)
    try:
        # parsed = json.loads(json_str)
        # print("Asdasd",parsed)
        message_content = json_str.get("messages", [{}])[0].get("content", None)
        return message_content
    except (json.JSONDecodeError, KeyError, TypeError):
        # print("Asdasasdd",json.JSONDecodeError)
        return None
    
def get_original_spans_df():
    span_df = px.Client().get_spans_dataframe(project_name="default")
    span_df["attributes.llm.output_messages"] = pd.json_normalize(span_df["attributes.llm.output_messages"])[0].to_list()
    span_df["attributes.llm.input_messages"] = pd.json_normalize(span_df["attributes.llm.input_messages"])[0].to_list()
    return span_df

def get_spans_df():
    # span_df = px.Client().get_spans_dataframe(project_name="default")
    # span_df["attributes.llm.output_messages"] = pd.json_normalize(span_df["attributes.llm.output_messages"])[0].to_list()
    # span_df["attributes.llm.input_messages"] = pd.json_normalize(span_df["attributes.llm.input_messages"])[0].to_list()
    query = SpanQuery().select(
      input="input.value",
      output="output.value",
    )

    final_res =  px.Client().query_spans(query)

    final_res["latency"], final_res["total_tokens"], final_res["model"], final_res["output_message"] = zip(
    *final_res['output'].apply(process_output_json_column)
    )

    final_res["input_message"] = final_res['input'].apply(process_input_json_column)

    final_res = final_res.drop(['input', 'output'], axis=1)

    return final_res

LLM_EVALUATOR_TEMPLATE = """
You are an evaluator. Your job is to decide if the provided answer is a valid response to the question.

**Your instructions:**
1. Carefully analyze the question to understand its intent.
2. Examine the answer to determine whether it satisfies the intent of the question.
3. Provide your reasoning step-by-step to justify your decision.
4. Output your reasoning in the format strictly provided below:
   - Start with "EXPLANATION:" followed by your reasoning in one or two sentences.
   - End with "LABEL:" followed by either "VALID" or "INVALID" (in uppercase, without quotes or punctuation).

**Important Guidelines:**
- Do not change the output format.
- Do not provide extra information, summaries, or comments outside the specified format.
- The output must consist only of the explanation and label in the specified format.

**Instructions on when to give INVALID:**
1. You can use your knowledegd and if the answer is false you give INVALID.
2. If you think answer is Hallucination give INVALID
3. If answer contains content which give you sense that actual answer is not provided or there is lack in knowledged or denial to answer give INVALID.

### Input:
Question:
{question}

Answer:
{answer}

### Expected Output Format:
EXPLANATION: [Your reasoning here.]
LABEL: [VALID or INVALID]

### Example Responses:
**Example 1**:
EXPLANATION: The answer is valid because the question asks for a definition of AI, and the answer provides a clear definition of AI.
LABEL: VALID

**Example 2**:
EXPLANATION: The answer is invalid because the question asks for an explanation about gravity, but the answer discusses photosynthesis instead.
LABEL: INVALID

### Task:
Evaluate the input using the above instructions and respond strictly in the required format.
"""

def evaluate_row(row, model, LLM_EVALUATOR_TEMPLATE):
  question = row['attributes.input.value']
  answer = row['attributes.output.value']
  chat_completion = client.chat.completions.create(
                    messages=[{
                              "role": "user",
                              "content": LLM_EVALUATOR_TEMPLATE.format(question=question, answer=answer),
                              }],
                    model=model,
                  )
  explanation, label = chat_completion.choices[0].message.content.split("LABEL")
  if "INVALID" in label:
    label = "INVALID"
  else:
    label = "VALID"
  return explanation, label


def evaluate_model(model, LLM_EVALUATOR_TEMPLATE=LLM_EVALUATOR_TEMPLATE):
  query = SpanQuery().where("annotations['Response Format'].label == None")
  df =  px.Client().query_spans(query)
  df['explanation'], df['label'] = zip(*df.apply(lambda row: evaluate_row(row, model, LLM_EVALUATOR_TEMPLATE), axis=1))
  df['score'] = df['label'].apply(lambda x: 1 if x == 'VALID' else 0)
  px.Client().log_evaluations(SpanEvaluations(eval_name="Response Format", dataframe=df))
  df = df[['attributes.output.value', 'attributes.input.value', 'explanation','label','score']]

  df["latency"], df["total_tokens"], df["model"], df["output_message"] = zip(
    *df['attributes.output.value'].apply(process_output_json_column)
  )

  df["input_message"] = df['attributes.input.value'].apply(process_input_json_column)

  df = df.drop(['attributes.output.value', 'attributes.input.value'], axis=1)

  return df

def get_dataset(name):
  dataset = px.Client().get_dataset(name=name)
  return dataset

def dataEvalResults(model, df):
  eval_model = LiteLLMModel(model=f"groq/{model}")

  hallucination_evaluator = HallucinationEvaluator(eval_model)
  qa_evaluator = QAEvaluator(eval_model)

  df["reference"] = df["metadata"]
  assert all(column in df.columns for column in ["output", "input", "reference"])

  hallucination_eval_df, qa_eval_df = run_evals(
      dataframe=df, evaluators=[hallucination_evaluator, qa_evaluator], provide_explanation=True
  )

  results_df = df.copy()
  results_df["hallucination_eval"] = hallucination_eval_df["label"]
  results_df["hallucination_explanation"] = hallucination_eval_df["explanation"]
  results_df["qa_eval"] = qa_eval_df["label"]
  results_df["qa_explanation"] = qa_eval_df["explanation"]

  results_df["output_message"] = df['output'].apply(process_json_column_with_data)
  results_df["input_message"] = df['input'].apply(process_json_column_with_data)

  results_df = results_df.drop(['output', 'input', 'metadata', 'reference'], axis=1)
  
  return results_df

def generate_answer(question, answer, LLM_MODEL):
  content = question.get("messages", [{}])[0].get("content", None)
  response = client.chat.completions.create(
      model=LLM_MODEL,
      messages=[
          {"role": "system", "content": "You are helpful agent and you give all answer to user input if you dont now you say. I dont know."},
          {"role": "user", "content": content},
      ],
  )
  cur_llm_ans = response.choices[0].message.content
  saved_llm_ans = answer.get("messages", [{}])[0].get("content", None)

  embedding1 = SentenceTransformer_model.encode(cur_llm_ans, convert_to_tensor=True)
  embedding2 = SentenceTransformer_model.encode(saved_llm_ans, convert_to_tensor=True)

  cosine_sim = util.cos_sim(embedding1, embedding2)

  return {"result":cur_llm_ans, "score": cosine_sim.item()}

def expected_output(output) -> bool:
    return output["score"] > 0.5

def task(input, expected, LLM_MODEL):
    return generate_answer(input, expected, LLM_MODEL)

def process_json_column_with_exper(json):
  try:
      # parsed = json.loads(json_str)
      # print("Asdasd",parsed)
      result = json.get("result", None)
      score = json.get("score", None)
      return result, score
  except (json.JSONDecodeError, KeyError, TypeError):
        # print("Asdasasdd",json.JSONDecodeError)
      return None, None

def modelExperiment(model, dataset):
  experiment = run_experiment(dataset, task=lambda input, expected: task(input, expected, LLM_MODEL=model), evaluators=[expected_output])

  results_df = experiment.as_dataframe()

  results_df["expected_output"] = results_df['expected'].apply(process_json_column_with_data)

  results_df["input_message"] = results_df['input'].apply(process_json_column_with_data)

  results_df["model_output"], results_df["similarity_score"]  = zip(*results_df['output'].apply(process_json_column_with_exper))
  

  results_df = results_df.drop(['expected', 'input', 'metadata', 'output'], axis=1)  

  return results_df