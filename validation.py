import streamlit as st
import pandas as pd
import openai
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
# Step 1: Set up Streamlit app to upload file
st.title('Data Validation App')

openai_api_key = st.sidebar.text_input('OpenAI API Key', value='', type='password')
if openai_api_key == 'bumin':
    openai_api_key = st.secrets['OPENAI_API_KEY']

uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

def check_columns(external_result, description_result):
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", """When given two pieces of information, Identify any mismatches or missing information. Be :\n\n"""),
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


    return output



def check_columns(external_result, description_result):
        prompt = f"Compare the following two pieces of information:\n\nExternal Result: {external_result}\nDescription Result: {description_result}\n\nIdentify any mismatches or missing information."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50
        )
        return response.choices[0].text.strip()

if uploaded_file:
    # Load the uploaded Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)
    
    st.write("File Uploaded Successfully!")
    st.write(df.head())

    # Step 2: Cross-check '외부결과' and '서술결과'
     # Replace with your OpenAI API key
    
    

    mismatches = []

    for index, row in df.iterrows():
        external = row['외부결과']
        opinion = row['서술결과']
        
        result = check_columns(external, opinion)
        
        if "no mismatch" not in result.lower():
            mismatches.append({
                'No': row['No'],
                '차트번호': row['차트번호'],
                '성명': row['성명'],
                'Comment': result
            })

    # Step 3: Display results in the Streamlit app
    if mismatches:
        st.write("Mismatches or missing information found:")
        mismatch_df = pd.DataFrame(mismatches)
        st.write(mismatch_df)
    else:
        st.write("No mismatches or missing information found.")