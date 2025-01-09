from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
import os 
import streamlit as st

os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
#os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
#os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
#os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']


llm4o = ChatOpenAI(model="gpt-4o", temperature=0)

generator_system_prompt = """You are a useful medical assitant that helps write medical notes.
Given a list of positive findings in lab results, radiological reports or other medical reports, summarize and write brief recommendations.
the final report should look like the example below.
[example]
00372955 김순례 [240920 HPDP]
pmhx: Asthma, Allergic rhinitis, HTN, Hypothyroidism, DL, DM 
current medication: Asthma, Allergic rhinitis medi (흡입제), HTN, Hypothyroidism, DL, DM medi 
op/adm hx: s/p cervical/lumbar disc herniation op
FHx: HTN/DM/Ca/CVD (-/-/-/+; 동생-MI) 

** Urine & Blood tests
#Urine epi-cell 2-5, Urine hyaline cast
  -Rec) UA recheck
#R.O preDM, R.O DM (A1c 6.5, FBS 118)
  -local에서 f/u 중이라고 하심. 
  -P) 다니시는 병원에서 지속 진료
#Dyslipidemia (Total chol 271, Calculated LDL 168, TG 272)
  -local에서 f/u 중이라고 하심.
  -P) 다니시는 병원에서 지속 진료
#mild hyperkalemia (K 5.38)
  -maybe d/t dehydration followed by CFS bowel prep
  -Associated Sx 
  -Rec) obs or electrolyte f.u 

**Imaging w/u 
#EGD: EG, RE (Bx: CG w/ healing erosion)
#CFS: C. diverticulosis, polyps x6 (bx: TALG x2, HP x3, IP x1)
  -Rec) 2YL CFS f.u
#LDCT
  - R.O Chronic bronchiolitis or NTM lung disease
    >> Rec) PU visit if Sx + (현재 천식 관련해 내과 진료 중이라고 하시어, 해당 병원에서 f/u 하시도록 설명드림.)
  -3mm pul. nodule (Lung-RADS C2, no change)
   >> Rec) 12ML LDCT f.u
  -mild degenerative spondylosis, diffuse osteoporosis, compression deformity at T6 and L1 
    >> 검진 당일 함께 시행한 BMD: T-score [L1-L2] -2.6
   >> P) EC or OS refer for medical tx. for osteoporosis
#Upper abdomen sono: R.O exophytic hepatic cyst (3.37cm, increased size) 
  -P) GE refer for further w.u
#Thyroid sono: slightly heterogenous echogenicity (TFT wnl)
#Carotid sono: 2.4mm plaque at Lt. carotid bulb
  -Rec) LSM, 6ML sono f.u
#Mammogram: focal asymmetry in Lt. breast central area (BI-RADS C0)
  -P) GS refer for Breast sono & Spot compression view
#CXR: wnl

**Others
#BMI: overweight
[end of example]

Here are some additional rules:
- only keep the titles of pmhx, op/adm hx, FHx if it is not mentioned.
- Group related lab results in the same line.
- Only list the results given in the reports. Don't write any test results that are not listed
- Only answer based on the information given by the user. NEVER make up anything. It's fine to leave some the content out, just try your best.
- for non lab results, write the name of the test first. ex) EGD, CFS, LDCT, Upper abdomen sono, Thyroid sono, Carotid sono, Mammogram, CXR, BMI
- for reports that mention any clinical correlation please give a list of symptoms that should be checked. ex) -Cardiovascular sx(chest pain, SOB, palpitations, syncope, edema, claudication : -/-/-/-/-/-)
"""
generator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generator_system_prompt),
        ("human", "medical test results: {report}"),
    ]
)
generator = generator_prompt |llm4o | StrOutputParser()
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the medical test results, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = llm4o.with_structured_output(GradeHallucinations)

hallucination_system_prompt = """You are a grader assessing whether an LLM generation is grounded in a set of medical test results.\n\nGive a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in supported by the set of medical test results."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_system_prompt),
        ("human", "Set of results: {test_results}\n\nLLM generation: {response}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader




def generate(state):
    """
    Validates whether the given explanation matches the test report.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    input = state["input"]
    
    response = generator.invoke({'report':input})
    return {"response": response}
  
    
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    input = state['input']
    response = state["response"]

    score = hallucination_grader.invoke(
        {"test_results": input, "response": response}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        return "useful"
    else:
        return "not useful"
    

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        test_report: test report
        explanation: original explanation of test report
        comment: comment of why the explanation doesn't match the test report
        new_explanation: new explanation of the test report
    """

    input: str
    response : str

workflow = StateGraph(GraphState)
# Define the nodes

workflow.add_node("generate", generate)   

# Build graph

workflow.add_edge(START, "generate")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "not useful": "generate",
    },
)
# Compile
autoformat = workflow.compile()

