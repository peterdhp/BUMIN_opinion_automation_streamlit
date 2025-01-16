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

박유경 [241211 HPDP]
pmhx
current medi
op/adm hx
FHx



**Urine & Blood tests
#Urine protein 1+, Urine epi-cell 2-5, yeast like cell, Hyaline cast
  -Rec) UA recheck

#Total chol 217, HDL 87
  -Rec) LSM

#WBC 10.49 x 10^3 (Neutrophil 83.3%), CRP 1.42
  -Associated sx ( )
  -Rec) CBC & Inflammatory marker f/u, Consider further w.u to idenfiy the cause if the figures are elevated continuously
            >> Refer 해드린 내과 분과 (NE) 통해 혈액검사 재검해보시도록 권고 드림. 



**Endoscopy
#EGD: CSG
  -Rec) EGD f.u within 2 years

#CFS: polyp x1 (bx: HP x1), Colon diverticulum, IH
  -Rec) CFS f.u within 3 years



**Imaging w/u
#Mammogram: BI-RADS C1, Severely dense breast (4)
  -Rec) Routine breast exam (Mammogram & Breast sono)

#Thyroid sono: mutiple colloid cysts in both lobe (K-TIRADS 2), 7.4mm iso-hyperechoic nodule in Rt. lobe (K-TIRADS 3), 0.23 x 0.34cm taller than wide marked hyperechoic nodule in Rt. lobe (K-TIRADS 4)
  -Rec) 6ML sono f.u

#Lower abdomen sono: WNL in observation (Limited evaluation of bladder)

#Upper abdomen sono: 1.23cm hyperechoic nodular lesion in Lt. lobe (R.O Hepatic hemangioma), Accessory spleen
  -Rec) GE refer for hepatic nodular lesion for either further w.u or regular sono f.u



**Others
#Pap: (-); inflammation
  -Rec) Routine screening

#Fundography: R.O Glaucoma (OD)
  -Rec) OPH visit
  
[end of example]

Here are some additional rules:
- only keep the titles of pmhx, op/adm hx, FHx if it is not mentioned.
- Group related lab results in the same line.
- Only list the results given in the reports. Don't write any test results that are not listed
- Only answer based on the information given by the user. NEVER make up anything. It's fine to leave some the content out, just try your best.
- for non lab results, write the name of the test first. ex) EGD, CFS, LDCT, Upper abdomen sono, Thyroid sono, Carotid sono, Mammogram, CXR, BMI
- for lab results or test results that mention clinical correlation please give a list of symptoms that should be checked. Here is an example of the line to be added. -Cardiovascular sx(chest pain, SOB, palpitations, syncope, edema, claudication : -/-/-/-/-/-)
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

