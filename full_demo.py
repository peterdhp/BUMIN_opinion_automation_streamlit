import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from streamlit_feedback import streamlit_feedback
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.tracers.context import collect_runs
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableAssign
)
from operator import itemgetter
import os
from langchain.vectorstores import FAISS
import cohere
from langchain_openai.embeddings import OpenAIEmbeddings
from langsmith import traceable
from langsmith import Client


os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]=st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"]=st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']

openai_api_key = st.sidebar.text_input('OpenAI API Key', value = '',type='password')
if openai_api_key =='bumin':
    openai_api_key = st.secrets['OPENAI_API_KEY']

client = Client()

#openai_api_key = os.environ.get('OPENAI_API_KEY')

### chroma는 semantic chunking, chroma_recursivesplit은 recursivecharactersplit으로 chunking한 것입니다.

def format_docs(docs: List[Document]) -> str:
    formatted = [
        f"{doc.metadata['korean']}"
        for doc in docs
    ]
    return "\n" + "\n".join(formatted)

def retrieve_and_merge(medical_record :str) -> list[Document] :
    FAISS_PATH = 'faiss'
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.load_local(folder_path=FAISS_PATH,embeddings=embeddings,index_name ="opinion_results_english", allow_dangerous_deserialization=True)
    ### k를 통해 몇개의 문서를 retrieve 해올지를 정할 수 있습니다.
    retriever= db.as_retriever(search_kwargs={'k':20, 'fetch_k':40})
    docs_list = retriever.invoke(medical_record)
    docs = {x.page_content: i for i, x in enumerate(docs_list)}
    rerank_input = list(docs.keys())
    
    return docs_list, rerank_input

def rerank(rerank_input,test_results):
    #print(rerank_input.keys())
    
    co = cohere.Client()
    rerank_response = co.rerank(
        query=test_results, documents= rerank_input, top_n=10, model="rerank-english-v3.0"
    )
    
    return rerank_response

def compress_retrieve(test_results):
    
    unique_docs, rerank_input = retrieve_and_merge(test_results)
    rerank_response = rerank(rerank_input,test_results)
    docs = [unique_docs[i.index] for i in rerank_response.results]
    
    return docs

def opinion_generator(model):
    
    opinion_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a doctor at a health screening center. Explain the test results to the patients. 
Use Korean. Choose one of the templates and complete it using the given results. Only use the information given in the test results. NEVER make up new information.  
Only output the result. However, when a biopsy was performed but doesn't have any ㅔathology diagnosis, please add a message at the end requesting more information: 
{template}"""),
    ("user", """
<test results>
{test_results}""")
    
])

    llm = ChatOpenAI(model_name=model, temperature = 0)  #gpt-4-turbo-preview #gpt-3.5-turbo #gpt-4 #gpt-3.5-turbo-1106
    output_parser = StrOutputParser()

    opinion_chain = opinion_prompt | llm | output_parser
    
    chain = (RunnableParallel(
            test_results=RunnablePassthrough(),
            docs= RunnableLambda(compress_retrieve), # Ensure 'retriever' is defined and configured properly
        ).assign(template=itemgetter("docs") | RunnableLambda(format_docs))
        .assign(opinion=opinion_chain)
        .pick(['opinion','docs'])
    )
    

       
    return chain
    

    
if "result_report" not in st.session_state:
    st.session_state.result_report = ''
if "opinion" not in st.session_state:
    st.session_state.opinion = {}
    
    
st.title('판정 소견 자동 완성')
feedback_option = "thumbs"


with st.form('my_form'):
    result_report = st.text_area('Enter text:', placeholder='submit test results here', height=400)
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        chain = opinion_generator(model="gpt-4o")
        with collect_runs() as cb:
            response = chain.invoke(result_report)
            st.session_state.run_id = cb.traced_runs[0].id
        opinion = response['opinion']
        doc_list = '\n\n'.join([f"{entry.metadata['test']}  :  {entry.metadata['korean']}" for i, entry in enumerate(response['docs'])])
        st.session_state.opinion = st.info(opinion)
        if st.session_state.get("run_id"):
            run_id = st.session_state.run_id
            feedback = streamlit_feedback(
                feedback_type=feedback_option,
                optional_text_label="[Optional] Please provide an explanation",
                key=f"feedback_{run_id}",
            )

            # Define score mappings for both "thumbs" and "faces" feedback systems
            score_mappings = {
                "thumbs": {"👍": 1, "👎": 0},
                "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
            }

            # Get the score mapping based on the selected feedback option
            scores = score_mappings[feedback_option]

            if feedback:
                # Get the score from the selected feedback option's score mapping
                score = scores.get(feedback["score"])

                if score is not None:
                    # Formulate feedback type string incorporating the feedback option
                    # and score value
                    feedback_type_str = f"{feedback_option} {feedback['score']}"

                    # Record the feedback with the formulated feedback type string
                    # and optional comment
                    feedback_record = client.create_feedback(
                        run_id,
                        feedback_type_str,
                        score=score,
                        comment=feedback.get("text"),
                    )
                    st.session_state.feedback = {
                        "feedback_id": str(feedback_record.id),
                        "score": score,
                    }
                else:
                    st.warning("Invalid feedback score.")
        
        
        
        st.info(doc_list)