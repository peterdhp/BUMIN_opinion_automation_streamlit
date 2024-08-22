import streamlit as st
import pdfplumber
import pandas as pd
import re
from menu_streamlit import menu_with_redirect

if 'result_report' not in st.session_state:
    st.session_state.result_report = ''

st.title("검진 결과 업로드하기")
file = st.file_uploader("검진 결과 기록지를 업로드해주세요.", type=["pdf"])

menu_with_redirect()

def clean_extracted_text(extracted_text):
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', extracted_text)
    
    # Split back into lines at reasonable places (e.g., punctuation)
    lines = re.split(r'(?<=[.!?]) +', cleaned_text)
    return '\n'.join(lines)

def extract_text_and_tables(pdf_path):
    full_transcript = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract and clean page text
            text = page.extract_text()
            if text:
                cleaned_text = clean_extracted_text(text)
                full_transcript += cleaned_text + "\n"
            
            # Extract tables and add them without any modification
            tables = page.extract_tables()
            for table in tables:
                # Convert table to a DataFrame
                df = pd.DataFrame(table[1:], columns=table[0])
                # Convert DataFrame to a Markdown string
                table_str = df.to_markdown(index=False)
                full_transcript += "\n" + table_str + "\n"
    
    return full_transcript

# Usage
with st.spinner("검진 결과를 읽고 있습니다."):
    st.session_state.result_report = extract_text_and_tables(file)
    st.switch_page("pages/consultation_chatbot.py")
