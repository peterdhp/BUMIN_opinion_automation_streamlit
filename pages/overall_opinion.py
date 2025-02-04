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
from langchain.vectorstores import FAISS
import cohere
from langsmith import traceable
from langsmith import Client

#os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
#os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
#os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
#os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']


menu_with_redirect()

client = Client()

def refine_overall_opinion(model):
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", """검진 결과를 보고 종합적으로 생활 습관을 어떻게 조정해야하는지, 어떠한 부분은 좋지 않아 진료가 필요한지, 혹시 위중하여 신속한 전문의의 진료가 필요한 경우는 없는지 나눠서 설명해줘. 정상인 결과들에 대해서도 요약해서 빠짐 없이 설명해주고, 내용을 지어내지 않도록 유의해줘. 의학용어는 괄호안에 아주 간단한 설명을 넣어줘. Don't use ** for header or subheaders, just ouput in plain text or use indents and -."""),
        ("user", """
[검진 결과]
{results}""")
    ])

    llm = ChatOpenAI(model_name=model, temperature=0)
    output_parser = StrOutputParser()

    refine_chain = refine_prompt | llm | output_parser


    return refine_chain

if "result" not in st.session_state:
    st.session_state.result = ''
if "overall_opinion" not in st.session_state:
    st.session_state.overall_opinion = ''

st.title('종합 소견 자동 완성')
feedback_option = "thumbs"
col1, col2 = st.columns(2)
with col1:
    with st.form('my_form'):
        result = st.text_area('Enter text:', placeholder='종합 소견을 입력해주세요', height=400)
        submitted = st.form_submit_button('Submit')
        if not st.session_state.openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and st.session_state.openai_api_key.startswith('sk-'):
            refine_chain = refine_overall_opinion(model="gpt-4o")
            with collect_runs() as cb:
                response = refine_chain.invoke(result)
                st.session_state.run_id = cb.traced_runs[0].id
            refined_opinion = response
            st.session_state.overall_opinion = refined_opinion 
            st.session_state.overall_opinion = st.session_state.overall_opinion + """\n\n ▶ 상기 종합소견은 검사 결과들에 대한 요약 설명입니다. 별도의 설명이 없는 검사들은 결과가 정상이거나 또는 참고치 범위를 약간 벗어나있는 경우이며, 이는 해당 결과값 단독으로는 임상적 의미가 없는 경우입니다. 검사 결과 확인 시 이점을 참고하시기 바랍니다.

    ▶ 질병의 조기발견 및 예방을 위하여 정기적인 검진을 권유 드립니다.

    ▶ 결과에 대하여 직접 상담을 원하시는 경우에는 상담 일정 예약 후 센터로 내원하시기 바랍니다.

    ▶ 결과 관련 문의 사항 : 02) 2620 - 0025"""
with col2:
    if not st.session_state.overall_opinion == '' :
        st.text_area('',value = st.session_state.overall_opinion,height=600)

    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id# Debug print for Run ID
        feedback = streamlit_feedback(
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

        if feedback:
            # Get the score from the selected feedback option's score mapping
            score = scores.get(feedback["score"])

            if score is not None:
                # Formulate feedback type string incorporating the feedback option and score value
                feedback_type_str = f"{feedback_option} {feedback['score']}"

                # Record the feedback with the formulated feedback type string and optional comment
                feedback_record = client.create_feedback(
                    run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Invalid feedback score.")
                
            