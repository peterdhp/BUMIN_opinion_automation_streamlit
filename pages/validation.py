import streamlit as st
import pandas as pd
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
# Step 1: Set up Streamlit app to upload file

from io import BytesIO
from menu_streamlit import menu_with_redirect
import os

os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']



menu_with_redirect()
st.title('Validation')

openai_api_key = st.sidebar.text_input('OpenAI API Key', value='', type='password')
if openai_api_key == 'bumin':
    openai_api_key = st.secrets['OPENAI_API_KEY']

uploaded_files = st.file_uploader("Upload your Excel file", type=['xlsx'], accept_multiple_files=True)

def check_columns(external_result, description_result):
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are going to be given a medical examination result report and a description of the result to inform the patient, Identify any mismatches or missing information. Be sensitive about the urgency mentioned.
If the description matched the result report, please out "no mismatch". If not, comment why it is mismatching in korean. only output the comment. Be as simple as possible.

This is an example that is considered matching. 
example Result Report : 
Finding
Procedure Note
Sedation: (Yes)
Level of sedation : moderate (paradoxical response: no)
Profopol : 20 mg, Midazolam 5 mg

Endoscopic finding
ESOPHAGUS : The mucosal blurring of Z-line.
STOMACH   : Diffuse mucosal atrophy with villous appearance was noticed on antrum and body.
                  : riased mucosal lesion was noticed on antrum-GC(Bx.A)
DUODENUM  : Non-specific finding.

Conclusion
- Reflux esophagitis, LA-minimal change
- Chronic atrophic gastritis & Intestinal metaplasia
- Gastric erosion

rec) EGD 1year f/u

example Description : 
* 위내시경 검사에서 역류성 식도염이 관찰되었습니다. 위산 역류를 악화시키는 음주, 흡연, 과식, 기름진 음식, 카페인 음료, 초콜릿 등을 피하시고 속쓰림, 흉부 불편감 등의 증상이 있는 경우 약물 치료를 받으시기 바랍니다. 경과 관찰을 위해 매년 정기적인 위내시경 검사를 받으시기 바랍니다.
* 위내시경 검사에서 위축성 위염 및 장상피화생 소견이 관찰되었습니다. 함께 시행 한 조직검사결과 만성 위염이 확인되었습니다. 위축성 위염은 위 점막의 위축성 변화이고, 장상피화생은 위 점막 상피의 불완전 재생에 의한 변화를 의미합니다. 경과관찰을 위해 1년 후 위내시경 검사를 받으시기 바랍니다.

         :\n\n"""),
        ("user", """
[Result Report]: 
{external_result}

[Description Report]: 
{description_result}
""")
    ])

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    output_parser = StrOutputParser()

    validation_chain = validation_prompt | llm | output_parser
    output = validation_chain.invoke({'external_result': external_result, 'description_result':description_result})

    return output




if uploaded_files:
    # Load the uploaded Excel file into a DataFrame
    patient_data = {}
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file)
        
        # Process each row of the file
        for index, row in df.iterrows():
            외부결과 = row['외부결과']
            서술결과 = row['서술결과']
            검사명칭 = row['검사명칭']
            성명 = row['성명']
            No = row['No']
            차트번호 = row['챠트번호']
            검진일 = row['검진일']

            result = check_columns(외부결과, 서술결과)
            
            # Create a unique identifier for each patient
            patient_key = (성명, 차트번호, 검진일)
            
            # If patient doesn't exist in the dictionary, initialize an entry
            if patient_key not in patient_data:
                patient_data[patient_key] = {
                    '성명': 성명,
                    '챠트번호': 차트번호,
                    '검진일': 검진일
                }
            
            # Add the test validation result as a new column based on '검사명칭'
            patient_data[patient_key][검사명칭] = result
              # Adding a delay to prevent rate limit issues

    # Convert the patient data dictionary to a DataFrame for displaying
    final_results = pd.DataFrame(patient_data.values())
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for test_date, group in final_results.groupby('검진일'):
            # Write each group (corresponding to a test date) into a separate sheet
            group.to_excel(writer, sheet_name=str(test_date), index=False)

    # Step 3: Display final results in the Streamlit app
    if not final_results.empty:
        st.download_button(
        label="Download Excel file with validation results",
        data=output,
        file_name="validation_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
        
    else:
        st.write("No mismatches or missing information found across the files.")