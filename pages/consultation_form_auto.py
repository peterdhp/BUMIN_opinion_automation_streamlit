import streamlit as st
import pandas as pd
import math

from pydantic import BaseModel, Field

from io import BytesIO
from menu_streamlit import menu_with_redirect
import os
from consulation_form_engine import autoformat as autoformat_chain

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
if "overall_text" not in st.session_state :    
    st.session_state.overall_text

openai_api_key = st.sidebar.text_input('OpenAI API Key', value='', type='password')
if openai_api_key == 'bumin':
    openai_api_key = st.secrets['OPENAI_API_KEY']

uploaded_files = st.file_uploader("Upload your Excel file", type=['xlsx'],accept_multiple_files=True)

def is_abnormal(row):
    result = str(row['검사결과'])  # Convert 검사결과 to string for consistent comparison
    reference = row['선택참고']
    
    # Check for non-numeric references
    if isinstance(reference, str) and ('음성' in reference or '양성' in reference):
        # Handle cases with exact matches or alternatives like "음성|약양성"
        ref_values = reference.split('|')
        return result not in ref_values
    else:
        # For numeric ranges, proceed with the range check
        try:
            result = float(row['검사결과'])  # Convert back to float for numeric comparison
            lower, upper = map(float, str(reference).replace('|', '~').split('~'))
            return result < lower or result > upper
        except (ValueError, TypeError):
            # If conversion fails, consider it non-numeric and skip
            return False



if uploaded_files:
    
    if st.session_state.processed_form == False:
        for uploaded_file in uploaded_files:
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
            chartnum = df['챠트번호'].iloc[0]
            name = df['성명'].iloc[0]
            checkup_date = df['검진일'].iloc[0]
            first_row = str(chartnum) + ' ' + name + ' [' + str(checkup_date) + 'HPDP]'
            result_rows.append(first_row)
            for _, row in df.iterrows():
                if row['type'] in [0, 1] and is_abnormal(row):
                    result_rows.append(f"{row['검사명칭']}: {row['검사결과']}")
                elif row['type'] == 2:
                    external_result = row['외부결과'] if not pd.isna(row['외부결과']) else row['서술결과']
                    result_rows.append(f"{row['검사명칭']}: {external_result}")

            # Concatenate results with line breaks
            st.session_state.output_form = "\n".join(result_rows)
            
            autotemplate = autoformat_chain.invoke({"input" : st.session_state.output_form})
            single_result = autotemplate['response']
            
            st.session_state.overall_text += single_result + '\n --------------------------------------\n\n\n\n'
            #st.write(result_rows)
            
        st.text(st.session_state.overall_text)
        
        
        st.session_state.processed_form =True
        
    if st.session_state.processed_form == True :
        if st.session_state.overall_text is not "":
            st.text(st.session_state.overall_text)
            st.download_button(
                label="Download Patient Results",
                data=st.session_state.overall_text,
                file_name='상담 양식.txt',
                mime='text/plain'
            )
        else :
            st.write("No results found")
