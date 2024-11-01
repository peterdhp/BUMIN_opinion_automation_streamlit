import streamlit as st
import pandas as pd
import math

from pydantic import BaseModel, Field

from io import BytesIO
from menu_streamlit import menu_with_redirect
import os
from validation_engine import validation as validation_chain
from validation_engine import validation_scope as validation_chain_scope

os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
#os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
#os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
#os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']



menu_with_redirect()
st.title('상담 양식 자동 작성')
if "processed_form" not in st.session_state :
    st.session_state.processed_form = False
if "output_form" not in st.session_state :
    st.session_state.output_form = ""

openai_api_key = st.sidebar.text_input('OpenAI API Key', value='', type='password')
if openai_api_key == 'bumin':
    openai_api_key = st.secrets['OPENAI_API_KEY']

uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

def is_abnormal(row):
    try:
        result = float(row['검사결과'])
        lower, upper = map(float, row['선택참고'].replace('|', '~').split('~'))
        return result < lower or result > upper
    except:
        return False


if uploaded_file:
    
    if st.session_state.processed_form == False:
        st.session_state.output_form = ""
        # Load the uploaded Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        df = df.replace({'_x000D_\n': '\n'}, regex=True)
        df = df.replace({'\\?': ''}, regex=True)

        # Function to check if result is abnormal
        
        result_rows = []
        
        # Filter out rows with '검사명칭' matching the given values
        excluded_tests = ['흉부X선(1차)','유전자검사','선헬스케어 동의서', '어떠케어 동의서', '에임메드 동의서','비플러스케어(becare) 동의서','케어링크 동의서+신분증사본','DHAT','위조직검사','위조직검사1~3개']
        
        # Assuming '검사명칭' is one of the columns. We need to filter based on that.
        df = df[~df['검사명칭'].isin(excluded_tests)]
        # Filter for a specific patient using their '챠트번호'

        for _, row in df.iterrows():
            if row['type'] in [0, 1] and is_abnormal(row):
                result_rows.append(f"{row['검사명칭']}: {row['검사결과']}")
            elif row['type'] == 2:
                if row['외부결과'] == '' :
                    result_rows.append(f"{row['검사명칭']}: {row['서술결과']}")
                else:
                    result_rows.append(f"{row['검사명칭']}: {row['외부결과']}")

        # Concatenate results with line breaks
        st.session_state.output_form = "\n".join(result_rows)
        st.session_state.processed_form =True

        

    st.text(st.session_state.output_form)