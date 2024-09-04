import streamlit as st
import pandas as pd
import openai
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
# Step 1: Set up Streamlit app to upload file
st.title('Validation')

openai_api_key = st.sidebar.text_input('OpenAI API Key', value='', type='password')
if openai_api_key == 'bumin':
    openai_api_key = st.secrets['OPENAI_API_KEY']

uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

def check_columns(external_result, description_result):
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are going to be given a medical examination result report and a description of the result to inform the patient, Identify any mismatches or missing information. Be sensitive about the urgency mentioned.
If the description matched the result report, please out "no mismatch". If not, comment why it is mismatching in korean.

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

    refine_chain = refine_prompt | llm | output_parser
    output = refine_chain.invoke({'external_result': external_result, 'description_result':description_result})

    return output




if uploaded_file:
    # Load the uploaded Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)
    
    #st.write("File Uploaded Successfully!")
    #st.write(df.head())

    # Step 2: Cross-check '외부결과' and '서술결과'
     # Replace with your OpenAI API key
    df = df[:5]
    

    mismatches = []

    for index, row in df.iterrows():
        external = row['외부결과']
        opinion = row['서술결과']
        
        result = check_columns(external, opinion)
        
        if "no mismatch" not in result.lower():
            mismatches.append({
                'No': row['No'],
                '차트번호': row['챠트번호'],
                '성명': row['성명'],
                'Comment': result
            })

    # Step 3: Display results in the Streamlit app
    if mismatches:
        st.write("Mismatches or missing information found:")
        mismatch_df = pd.DataFrame(mismatches)
        st.download_buttonl('result',mismatch_df)
    else:
        st.write("No mismatches or missing information found.")