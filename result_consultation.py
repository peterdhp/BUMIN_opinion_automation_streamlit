import streamlit as st
import os



os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]=st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"]=st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']

openai_api_key = st.secrets['OPENAI_API_KEY']



    

    
if "result_report" not in st.session_state:
    st.session_state.result_report = ''
    
    
st.title('건강검진 상담 챗봇')



with st.form('my_form'):
    result_report = st.text_area('검진 판정 결과 입력:', placeholder='판정결과를 입력해주세요', height=400)
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        st.session_state.result_report = result_report
        st.switch_page('pages/consultation_chatbot.py')