import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableAssign
)
import os

from langsmith import traceable
from menu_streamlit import menu_with_redirect

menu_with_redirect()


os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]=st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"]=st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']

openai_api_key = st.secrets['OPENAI_API_KEY']


if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "result_finalreport" not in st.session_state :
    st.switch_page("pages/result_submit.py")

#openai_api_key = os.environ.get('OPENAI_API_KEY')

### chroma는 semantic chunking, chroma_recursivesplit은 recursivecharactersplit으로 chunking한 것입니다.




def chat_engine(msg_log):
    system_prompt = [("system", """You are a Korean doctor. you are going to answer questions from a patient. Answer the question based on the examination results provided. :

[examination results]
{result_report}""")]
    
    prompt_temp = system_prompt + msg_log
    
    prompt = ChatPromptTemplate.from_messages(prompt_temp)
    
    llm = ChatOpenAI(model_name="gpt-4o", temperature = 0) 
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser    
    
    output = chain.stream({"result_report" : st.session_state.result_report})
    return output

for message in st.session_state.messages:
    role = '🩺' if message[0] == 'ai' else message[0]
    with st.chat_message(role):
        st.markdown(message[1])
        
        
        
if len(st.session_state.messages) ==0 : 
    #print(st.session_state.add_question)
    st.session_state.messages.append(('ai','안녕하세요? 판정 결과 도우미입니다. 건강 검진 판정결과에 대해서 궁금하신 것을 질문해주세요.'))
    
    with st.chat_message("🩺"):
        st.markdown('안녕하세요? 판정 결과 도우미입니다. 건강 검진 판정결과에 대해서 궁금하신 것을 질문해주세요.')

        
if userinput := st.chat_input("메시지를 입력해주세요."):
    # Add user message to chat history
    st.session_state.messages.append(("human", userinput))
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(userinput)
        
        
            
    
    # Display assistant response in chat message container
    with st.chat_message("🩺"):
        stream = chat_engine(st.session_state.messages)
        response = st.write_stream(stream)
    st.session_state.messages.append(("ai", response))
        
    