import streamlit as st
import pandas as pd
import math

from pydantic import BaseModel, Field

from io import BytesIO
from menu_streamlit import menu_with_redirect
import os
from validation_engine_2 import validation as validation_chain

os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
#os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
#os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
#os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']



menu_with_redirect()
st.title('Validation')
if "processed" not in st.session_state :
    st.session_state.processed = False
if "output_text" not in st.session_state :
    st.session_state.output_text = ""

openai_api_key = st.sidebar.text_input('OpenAI API Key', value='', type='password')
if openai_api_key == 'bumin':
    openai_api_key = st.secrets['OPENAI_API_KEY']

uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

if uploaded_file:
    
    if st.session_state.processed == False:
        st.session_state.output_text = ""
        # Load the uploaded Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        df = df.replace({'_x000D_\n': '\n'}, regex=True)
        df = df.replace({'\\?': ''}, regex=True)
        
        # Filter rows where 'type' is 2
        filtered_df = df[(df['type'] == 2) | (df['검사명칭'].isin(['유리갑상선호르몬(Free T4)', '갑상선자극호르몬(TSH)']))]
        filtered_df = filtered_df[~(filtered_df['검사결과'].isna() & filtered_df['외부결과'].isna())]
        #patient_chart_number = 332655  # Replace with the chart number of the patient you want to test
        #filtered_df = filtered_df[filtered_df['챠트번호'] == patient_chart_number]
        

        # Filter out rows with '검사명칭' matching the given values
        excluded_tests = ['체성분분석검사', '심전도검사', '동맥경화검사', '안저검사', '골밀도검사(spine)', '스트레스검사','흉부X선(1차)','유전자검사', '심장초음파'
                        '선헬스케어 동의서', '어떠케어 동의서', '에임메드 동의서','비플러스케어(becare) 동의서','케어링크 동의서+신분증사본','DHAT','위조직검사','위조직검사1~3개']
        
        # Assuming '검사명칭' is one of the columns. We need to filter based on that.
        filtered_df = filtered_df[~filtered_df['검사명칭'].isin(excluded_tests)]
        # Filter for a specific patient using their '챠트번호'
        
        ''''''

        # Group by patient and iterate through their tests
        
        for patient_name, patient_data in filtered_df.groupby(['성명', '챠트번호']):
            st.session_state.output_text += f"{patient_name[0]}    {patient_name[1]}\n"
            
            has_ultrasound = '유방초음파검사' in patient_data['검사명칭'].values
            has_xray = '유방X선검사' in patient_data['검사명칭'].values  
            has_thyroid= '갑상선초음파' in patient_data['검사명칭'].values
            has_TFT = '유리갑상선호르몬(Free T4)' in patient_data['검사명칭'].values
                
            if has_ultrasound and has_xray:
        # Concatenate the results for '외부결과' and '서술결과' for both tests
                combined_external_result = ''
                combined_narrative_result = ''
                
                for _, row in patient_data[patient_data['검사명칭'].isin(['유방초음파검사', '유방X선검사'])].iterrows():
                    combined_external_result += f"{row['외부결과']}\n" if pd.notna(row['외부결과']) else ''
                    combined_narrative_result = f"{row['서술결과']}\n" if pd.notna(row['서술결과']) else ''
                
                # Feed concatenated results into the custom function
                result = validation_chain.invoke({"test_report" : combined_external_result, "explanation" : combined_narrative_result})
                if 'comment' in result and 'new_explanation' in result:
                    st.session_state.output_text += f"유방초음파검사 + 유방X선검사 - {result['comment']}\n{result['new_explanation']}\n"
                elif 'comment' in result:
                    st.session_state.output_text += f"유방초음파검사 + 유방X선검사 - {result['comment']}\n"
                else:
                    st.session_state.output_text += f"유방초음파검사 + 유방X선검사 - 소견 일치\n"
                    
                patient_data = patient_data[~patient_data['검사명칭'].isin(['유방초음파', '유방x선검사'])]
                
                    
            
            if has_thyroid and has_TFT:
                US_row = patient_data[patient_data['검사명칭']=='갑상선초음파'].iloc[0]
                FT4_value = patient_data[patient_data['검사명칭']=='유리갑상선호르몬(Free T4)'].iloc[0]['검사결과']
                TSH_value = patient_data[patient_data['검사명칭']=='갑상선자극호르몬(TSH)'].iloc[0]['검사결과']
                combined_narrative_result = US_row['서술결과']
                combined_external_result = US_row['외부결과'] + f"\n\n Free T4 : {FT4_value} (Normal value : 0.92~1.68)  TSH : {TSH_value} (Normal value : 0.27~4.20)"
                    
                result = validation_chain.invoke({"test_report" : combined_external_result, "explanation" : combined_narrative_result})
                if 'comment' in result and 'new_explanation' in result:
                    st.session_state.output_text += f"갑상선초음파 - {result['comment']}\n{result['new_explanation']}\n"
                elif 'comment' in result:
                    st.session_state.output_text += f"갑상선초음파 - {result['comment']}\n"
                else:
                    st.session_state.output_text += f"갑상선초음파 - 소견 일치\n"
                    
                patient_data = patient_data[~patient_data['검사명칭'].isin(['갑상선초음파', '유리갑상선호르몬(Free T4)','갑상선자극호르몬(TSH)'])]
            else :
                patient_data = patient_data[~patient_data['검사명칭'].isin(['유리갑상선호르몬(Free T4)','갑상선자극호르몬(TSH)'])]
                    
            
                # Process normally for other tests of the patient
            for _, row in patient_data.iterrows():
                result = validation_chain.invoke({"test_report" : row['외부결과'], "explanation" : row['서술결과']})
                
                if 'comment' in result and 'new_explanation' in result:
                    st.session_state.output_text += f"{row.get('검사명칭', 'Unknown Test')} - {result['comment']}\n{result['new_explanation']}\n\n"
                elif 'comment' in result:
                    st.session_state.output_text += f"{row.get('검사명칭', 'Unknown Test')} - {result['comment']}\n"
                else:
                    st.session_state.output_text += f"{row.get('검사명칭', 'Unknown Test')} - 소견 일치\n"
            
            # Separator between different patients
            st.session_state.output_text += "-------------------------------------------\n"
        st.session_state.processed =True

    # Step 3: Display final results in the Streamlit ap
    if st.session_state.processed ==True :
        if st.session_state.output_text is not "":
            st.text(st.session_state.output_text)
            st.download_button(
                label="Download Patient Results",
                data=st.session_state.output_text,
                file_name='patient_results_single_patient.txt',
                mime='text/plain'
            )
        else :
            st.write("No results found")
        