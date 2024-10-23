import streamlit as st
import pandas as pd
import math

from pydantic import BaseModel, Field

from io import BytesIO
from menu_streamlit import menu_with_redirect
import os
from validation_engine import validation as validation_chain

os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']



menu_with_redirect()
st.title('Validation')
processed = False

openai_api_key = st.sidebar.text_input('OpenAI API Key', value='', type='password')
if openai_api_key == 'bumin':
    openai_api_key = st.secrets['OPENAI_API_KEY']

uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

if uploaded_file:
    
    if processed == False:
        output_text = ""
        # Load the uploaded Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        
        # Filter rows where 'type' is 2
        filtered_df = df[df['type'] == 2]
        patient_chart_number = 332655  # Replace with the chart number of the patient you want to test
        filtered_df = filtered_df[filtered_df['챠트번호'] == patient_chart_number]
        

        # Filter out rows with '검사명칭' matching the given values
        excluded_tests = ['체성분분석검사', '심전도검사', '동맥경화검사', '안저검사', '골밀도검사(spine)',
                        '선헬스케어 동의서', '어떠케어 동의서', '에임메드 동의서','비플러스케어(becare) 동의서','케어링크 동의서+신분증사본']
        
        # Assuming '검사명칭' is one of the columns. We need to filter based on that.
        filtered_df = filtered_df[~filtered_df['검사명칭'].isin(excluded_tests)]
        # Filter for a specific patient using their '챠트번호'
        
        ''''''

        # Group by patient and iterate through their tests
        
        for patient_name, patient_data in filtered_df.groupby(['성명', '챠트번호']):
            output_text += f"{patient_name[0]}            {patient_name[1]})\n"
            for _, row in patient_data.iterrows():
                output_text += row['검사명칭'] + "\n"
                
                result = validation_chain.invoke(row['외부결과'], row['서술결과'])
                print(result)
                if 'comment' in result and 'new_explanation' in result:
                    output_text += f"{row.get('검사명칭', 'Unknown Test')} - {result['comment']}\n{result['new_explanation']}\n"
                elif 'comment' in result:
                    output_text += f"{row.get('검사명칭', 'Unknown Test')} - {result['comment']}\n"
                else:
                    output_text += f"{row.get('검사명칭', 'Unknown Test')} - 소견 일치\n"
                
            # Separator between different patients
            output_text += "-------------------------------------------\n"
        processed =True

    # Step 3: Display final results in the Streamlit ap
    if processed ==True :
        if output_text is not "":
            st.write(output_text)
            st.download_button(
                label="Download Patient Results",
                data=output_text,
                file_name='patient_results_single_patient.txt',
                mime='text/plain'
            )
        else :
            st.write("No results found")
        