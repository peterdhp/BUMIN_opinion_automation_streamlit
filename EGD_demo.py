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


os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]=st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"]=st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']

openai_api_key = st.sidebar.text_input('OpenAI API Key', value = '',type='password')
if openai_api_key =='bumin':
    openai_api_key = st.secrets['OPENAI_API_KEY']


#openai_api_key = os.environ.get('OPENAI_API_KEY')

### chroma는 semantic chunking, chroma_recursivesplit은 recursivecharactersplit으로 chunking한 것입니다.



@traceable
def opinion_generator(model):
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a doctor at a health screening center. Explain the EGD results to the patients. 
Use Korean. Choose one of the templates and complete it using the given results: 
* 위내시경 검사에서 십이지장염이 관찰되었습니다. 금연, 금주와 함께 자극이 적은 부드러운 식사를 하십시오. 증상이 있는 경우 약물치료를 받으시기 바랍니다.
* 위내시경 검사에서 십이지장 궤양이 관찰됩니다. 궤양에 대해 치료를 받으시고 금연, 금주와 함께 자극이 적은 부드러운 식사를 하십시오.
* 위내시경 검사에서 십이지장 궤양이 관찰되고 헬리코박터균 반응에서 양성입니다. 제균치료 및 궤양에 대해 치료를 받으시고 제균여부에 대해 확인검사를 받으십시오. 또한 (금연, 금주와 함께) 자극이 적은 부드러운 식사를 하십시오.
* 위내시경 검사에서 십이지장 궤양의 반흔이 관찰됩니다. 현재 활동성 궤양은 없으나 추후 재발할 가능성이 있으므로 금연, 절주하시고 자극적인 음식을 피하시기 바랍니다. 경과 관찰을 위해 1년 후 위내시경 검사를 받으십시오. 
* 위내시경 검사에서 십이지장 궤양의 반흔이 관찰되었고, 헬리코박터균 검사가 양성으로 나왔습니다. 헬리코박터균에 대해 제균 치료를 받으시고 제균 여부에 대해 확인검사를 받으시기 바랍니다.
* 위내시경 검사에서 십이지장 용종이 관찰되었습니다. 변화양상을 파악하기 위하여 정기적으로 내시경 검사를 받으시기 바랍니다.
* 위내시경 검사에서 작은 십이지장 점막하종양이 관찰되었습니다. 대부분 양성 소견이나 드물게 천천히 커지는 경우도 있습니다. 경과관찰을 위해 6개월 후 위내시경 검사를 받으시기 바랍니다.
* 위내시경 검사에서 cm 크기의 십이지장 점막하종양이 관찰되었습니다. 추가적인 상담을 위해 소화기내과 진료를 받으시기 바랍니다.
* 위내시경 검사 결과 정상입니다. 속쓰림, 소화불량 등의 증상이 없으면 1-2년 후 정기검사 받으시기 바랍니다.
* 위내시경 검사에서 역류성 식도염이 관찰되었습니다. 위산 역류를 악화시키는 음주, 흡연, 과식, 기름진 음식, 카페인 음료, 초콜릿 등을 피하시고 속쓰림, 흉부 불편감 등의 증상이 있는 경우 약물 치료를 받으시기 바랍니다. 경과 관찰을 위해 매년 정기적인 위내시경 검사를 받으십시오.
* 위내시경 검사에서 역류성 식도염 및 그와 관련된 보초 용종이 있습니다. 음주, 흡연, 과식, 기름진 음식, 카페인 음료, 초콜렛 및 야식을 피하십시오. 증상이 있는 경우 약물 치료를 받으시기 바랍니다.경 검사를 받으십시오.
* 위내시경 검사에서 바렛 식도가 의심됩니다. 음주, 흡연, 과식, 기름진 음식, 카페인 음료, 초콜릿 및 야식을 피하십시오. 증상이 있는 경우 소화기내과 진료를 받으시고 약물치료를 받으시기 바랍니다. 변화양상 관찰을 위해 정기적으로 위내시경 검사를 받으시기 바랍니다.는 경우 약물 치료를 받으시기 바랍니다.경 검사를 받으십시오.
* 위내시경 검사에서 식도 유두종이 관찰되었습니다. 악성이 아닌 양성 종양으로 치료는 필요치 않습니다.
* 위내시경 검사에서 식도 점막하 종양이 관찰되었습니다. 양성 종양으로 치료는 필요치 않으며 경과관찰을 위해 1년마다 위내시경 검사를 받으시기 바랍니다.
* 위내시경 검사에서 식도 점막하 종양이 관찰되었습니다. 추가 상담을 위해 소화기내과 진료를 받으시기 바랍니다.
* 위내시경 검사 결과 정상입니다. 속쓰림, 소화불량 등의 증상이 없으면 2년 후 정기검사 받으시기 바랍니다.
* 위내시경 검사에서 위염이 있습니다. 위험인자(흡연,음주,자극적 식사,스트레스 등)가 있다면 이의 관리에 노력하시고 속쓰림, 소화불량 등의 증상이 있는 경우 약물치료를 받으시기 바랍니다. 증상이 없으시면 1년 뒤 추적 위내시경 검사를 받으십시오.
* 위내시경 검사에서 만성 위축성 위염이 관찰됩니다. 위축성 위염은 위점막에 위축성 변화가 온 것입니다. 경과관찰을 위해 1년 후 위내시경 검사를 받으시기 바랍니다.
* 위내시경 검사에서 위염 및 만성 위축성 위염이 있습니다. 위험인자(흡연,음주,자극적 식사,스트레스 등)가 있다면 이의 관리에 노력하시고 속쓰림, 소화불량 등의 증상이 있는 경우 약물치료를 받으시기 바랍니다. 증상이 없으면 위험인자 관리와 함께 자극이 적은 부드러운 식사를 하십시오. 경과관찰을 위해 1년 마다 위내시경 검사를 받으시기 바랍니다.
* 위내시경 검사에서 위축성 위염 및 장상피화생 소견이 관찰되었습니다. 위축성 위염은 위 점막의 위축성 변화이고, 장상피화생은 위 점막 상피의 불완전 재생에 의한 변화를 의미합니다. 경과관찰을 위해 1년 후 위내시경 검사를 받으시기 바랍니다.
* 위내시경 조직검사에서 헬리코박터와 관련된 위염이 나왔습니다. 헬리코박터균 감염은 위염, 위궤양, 십이지궤양 및 위암과 연관성이 높습니다. 제균치료에 대해 가정의학과 진료를 받으십시오.
* 위내시경 검사에서 활동기의 위궤양이 관찰되었습니다. 술, 담배, 자극이 되는 약물을 피하고 자극이 적은 부드러운 식사를 하십시오. 약물치료와 단기 추적 위내시경 검사가 필요하므로 소화기내과 진료를 받으시기 바랍니다.
* 위내시경 검사에서 활동기의 위궤양이 관찰되었고 위에 헬리코박터균 검사도 양성으로 나와서 이에 대한 치료가 필요합니다. 약물치료와 추적 위내시경 검사가 필요하므로 소화기 내과 진료를 받으시기 바랍니다. 술, 담배, 자극이 되는 약물을 피하고 자극이 적은 부드러운 식사를 하십시오.
* 위내시경 검사에서 위궤양의 흔적이 관찰됩니다. 현재 활동성 궤양은 없으나 추후 재발할 가능성이 있으므로 금연, 절주하시고 자극적인 음식을 피하시기 바랍니다. 경과 관찰을 위해 1년 후 위내시경 검사를 받으십시오. 
* 위내시경 검사에서 위궤양의 흔적이 관찰되고 헬리코박터균에 감염된 상태입니다. 제균치료 (및 궤양에 대해 치료)를 받으시고 제균여부에 대해 확인검사를 받으십시오. 또한 금연, 금주와 함께 자극이 적은 부드러운 식사를 하십시오.
* 위내시경 검사에서 용종 의심 소견이 관찰되어 조직검사를 시행하였습니다. 조직검사에서 위저선 용종 (fundic gland polyp) 으로 확인되었습니다. 일반적으로 위저선종의 경우 악성화 가능성이 거의 없는 것으로 알려져 있습니다. 변화여부 확인을 위해 1년 뒤 추적 검사를 받아보시기 바랍니다.
* 위내시경 검사에서 작은 위용종 의심 소견이 관찰됩니다. 일반적으로 작은 위용종의 경우 악성화 가능성이 거의 없는 것으로 알려져 있습니다. 변화여부 확인을 위해 1년 뒤 추적 검사를 받아보시기 바랍니다.
* 위내시경 검사에서 작은 위 점막하종양이 관찰되었습니다. 대부분 양성 소견이나 드물게 천천히 커지는 경우도 있습니다. 경과관찰을 위해 6개월 후 위내시경 검사를 받으시기 바랍니다.
* 위내시경 검사에서 cm 크기의 위 점막하종양이 관찰되었습니다. 추가적인 상담을 위해 소화기내과 진료를 받으시기 바랍니다."""),
    ("user", """
<EGD results>
{results}""")
    
])



    llm = ChatOpenAI(model_name=model, temperature = 0)  #gpt-4-turbo-preview #gpt-3.5-turbo #gpt-4 #gpt-3.5-turbo-1106
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    opinion = chain.invoke(result_report)        
    return opinion
    

    
if "result_report" not in st.session_state:
    st.session_state.result_report = ''
if "opinion" not in st.session_state:
    st.session_state.opinion = {}
    
    
st.title('내시경 소견 자동 완성')



with st.form('my_form'):
    result_report = st.text_area('Enter text:', placeholder='submit test results here', height=400)
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        opinion = opinion_generator(model="gpt-4o")
        st.session_state.opinion = st.info(opinion)