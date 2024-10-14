import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from streamlit_feedback import streamlit_feedback
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.tracers.context import collect_runs
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableAssign
)

from menu_streamlit import menu_with_redirect
from operator import itemgetter
import os
from langchain_community.vectorstores import FAISS
import cohere
from langsmith import traceable
from langsmith import Client

os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']


menu_with_redirect()

client1 = Client()
client2 = Client()


def opinion_generator_new(model):
    opinion_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a doctor at a health screening center. Explain the test results to the patients. Give brief general information about the diagnosis and a recommendation for the patient.
here is an example. 
input : non-specific T wave abnormalitiy
output : 전도 검사에서 비특이적 T 팡 이상 소견이 관찰됩니다. 허혈성 심질환 등에서 관찰될 수 있는 소견이나 정상인에서도 보일 수 있습니다. 흉통 등의 증상이 있는 경우 순환기내과 진료를 받으시기 바랍니다."""),
        ("user", """{diagnosis}

심""")
    ])

    llm = ChatOpenAI(model_name=model, temperature=0)
    output_parser = StrOutputParser()

    opinion_chain = opinion_prompt | llm | output_parser

    return opinion_chain

def multi_opinion_summary(model):
    opinion_prompt = ChatPromptTemplate.from_messages([
        ("system", """환자한테 설명하는 내용들이 여러개 있을 때 한 문단으로 자연스럽게 합쳐줘. 첫 문장은 'ㅇㅇㅇ 검사에서 ㅇㅇㅇ,ㅇㅇㅇ 및 ㅇㅇㅇ이 관찰됩니다.'으로 시작해서 설명을 덧붙이고 추적관찰, 어떤 과를 가라는 내용을 마지막에 적어줘. 되도록이면 새로운 내용을 만들어내진 말아줘."""),
        ("user", """{multi_results}

심""")
    ])

    llm = ChatOpenAI(model_name=model, temperature=0)
    output_parser = StrOutputParser()

    opinion_chain = opinion_prompt | llm | output_parser

    return opinion_chain

if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = ''
if "multi_opinion" not in st.session_state:
    st.session_state.multi_opinion = ''
if "opinion_new" not in st.session_state:
    st.session_state.opinion_new = ''
if "opinion_summary" not in st.session_state:
    st.session_state.opinion_summary = ''

if "doc_list" not in st.session_state:
    st.session_state.doc_list = ''

st.title('새로운 소견 자동 완성 및 여러 소견 합치기')
feedback_option = "thumbs"
col1, col2 = st.columns(2)
with col1 :
    with st.form('my_form'):
        diagnosis_new = st.text_input('Enter text:', placeholder='진단명 입력')
        submitted = st.form_submit_button('Submit')
        if not st.session_state.openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and st.session_state.openai_api_key.startswith('sk-'):
            chain = opinion_generator_new(model="gpt-4o")
            with collect_runs() as cb:
                response = chain.invoke(diagnosis_new)
                st.session_state.run_id = cb.traced_runs[0].id
            st.session_state.opinion_new = response # Store the opinion in the session state
    if not st.session_state.opinion_new == '' :
        st.info('*'+st.session_state.opinion_new)
        
    
    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id# Debug print for Run ID
        feedback_1 = streamlit_feedback(
            feedback_type=feedback_option,
            optional_text_label="코멘트를 입력해주세요.",
            key=f"feedback_{run_id}",
        )

        # Define score mappings for both "thumbs" and "faces" feedback systems
        score_mappings = {
            "thumbs": {"👍": 1, "👎": 0},
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        # Get the score mapping based on the selected feedback option
        scores = score_mappings[feedback_option]

        if feedback_1:
            # Get the score from the selected feedback option's score mapping
            score = scores.get(feedback_1["score"])

            if score is not None:
                # Formulate feedback type string incorporating the feedback option and score value
                feedback_type_str = f"{feedback_option} {feedback_1['score']}"

                # Record the feedback with the formulated feedback type string and optional comment
                feedback_record = client1.create_feedback(
                    run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback_1.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Invalid feedback score.")
            
            
            
with col2 :
    

    with st.form('my_form_2'):
        multi_opinion = st.text_area('Enter text:', placeholder='여러 소견 입력', height=400)
        submitted = st.form_submit_button('Submit')
        if not st.session_state.openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and st.session_state.openai_api_key.startswith('sk-'):
            chain = multi_opinion_summary(model="gpt-4o")
            with collect_runs() as cb:
                response = chain.invoke(multi_opinion)
                st.session_state.run_id = cb.traced_runs[0].id
            opinion = response
            st.session_state.opinion_summary = opinion # Store the opinion in the session state
    if not st.session_state.opinion_summary == '' :
        st.info('*'+st.session_state.opinion_summary)
        
    
    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id# Debug print for Run ID
        feedback_2 = streamlit_feedback(
            feedback_type=feedback_option,
            optional_text_label="코멘트를 입력해주세요.",
            key=f"feedback_{run_id}",
        )

        # Define score mappings for both "thumbs" and "faces" feedback systems
        score_mappings = {
            "thumbs": {"👍": 1, "👎": 0},
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        # Get the score mapping based on the selected feedback option
        scores = score_mappings[feedback_option]

        if feedback_2:
            # Get the score from the selected feedback option's score mapping
            score = scores.get(feedback_2["score"])

            if score is not None:
                # Formulate feedback type string incorporating the feedback option and score value
                feedback_type_str = f"{feedback_option} {feedback_2['score']}"

                # Record the feedback with the formulated feedback type string and optional comment
                feedback_record = client2.create_feedback(
                    run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback_2.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Invalid feedback score.")
            
        