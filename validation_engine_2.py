from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
import operator
import os 
import streamlit as st

os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
#os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
#os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
#os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']


llm4o = ChatOpenAI(model="gpt-4o", temperature=0)
llm4omini = ChatOpenAI(model="gpt-4o-mini", temperature=0)







class CheckBiopsy(BaseModel):
    """Binary score for whether a biopsy was performed according to the test report."""

    binary_score: str = Field(
        description="whether a biopsy was performed according to the test report. , 'yes' or 'no'"
    )
structured_biopsyChecker = llm4omini.with_structured_output(CheckBiopsy)

biopsyChecker_system_prompt = """You are a reviewer assessing whether a biopsy was performed according to the test report. 'Bx' is an abbreviation for biopsy.\n\nGive a binary score 'yes' or 'no'"""
biopsyChecker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", biopsyChecker_system_prompt),
        ("human", "Test Report:\n{report}"),
    ]
)
Biopsy_checker = biopsyChecker_prompt | structured_biopsyChecker



class CheckLimited(BaseModel):
    """binary score for whether the narrative explanation of a medical test correctly acknowledged any 'limitated evaluation' in the test’s evaluation."""

    binary_score: str = Field(
        description="Whether the narrative explanation of a medical test correctly acknowledged any 'limitated evaluation' in the test’s evaluation, 'yes' or 'no'"
    )
structured_limitCheck = llm4omini.with_structured_output(CheckLimited)

limitChecker_system_prompt = """You are a reviewer assessing a narrative explanation of a medical test. When the test report mentions 'limited evaluation', the explanation should mention it too. If it didn't say 'no'. Otherwise say 'yes'."""
limitChecker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", limitChecker_system_prompt),
        ("human", "Test Report:\n{report}\n\nNarrative explanation of test:\n{explanation}"),
    ]
)
limit_checker = limitChecker_prompt | structured_limitCheck




class AnomalyCheck(BaseModel):
    """List of positive findings or anomalities mentioned in the test report."""

    AnomalyList: List[str]= Field(
        description="list of positive findings or anomalities mentioned in the test report. If there are no positive findings or anomalities return an empty list."
    )
structured_anomalyCheck = llm4o.with_structured_output(AnomalyCheck)

anomalyChecker_system_prompt = """You are a medical reviewer assessing a test report. List up the positive findings or anomalities mentioned in the test report. Please omit redundant findings that are medically equivalent to what is written in the recommendations(rec) or impressions(impt). Be aware of medical abbreviatons."""
anomalyChecker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", anomalyChecker_system_prompt),
        ("human", "Test Report:\n{report}"),
    ]
)
anomaly_checker = anomalyChecker_prompt | structured_anomalyCheck




class NofindingsCheck(BaseModel):
    """List of positive findings or anomalities mentioned in the narrative explanation of a medical test."""

    AnomalyList: List[str] = Field(
        description="List of positive findings or anomalities mentioned in the narrative explanation of a medical test. If there are no positive findings or anomalities return an empty list."
    )
structured_nofindingsCheck = llm4o.with_structured_output(NofindingsCheck)

nofindingsChecker_system_prompt = """You are a medical reviewer assessing a test report. List up the positive findings or anomalities mentioned in the test report. Be aware of medical abbreviatons.\n\nIf there are no positive findings or anomalities return an empty list."""
nofindingsChecker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", nofindingsChecker_system_prompt),
        ("human", "Narrative explanation of test:\n{explanation}"),
    ]
)
nofindings_checker = nofindingsChecker_prompt | structured_nofindingsCheck




class CheckMentioned(BaseModel):
    """List of positive findings or anomalities that are not mentioned in the narrative explanation."""

    MissedFindings: List[str] = Field(
        description="List of positive findings or anomalities that are not mentioned in the narrative explanation. If there are none return empty list."
    )
structured_mentionChecker = llm4o.with_structured_output(CheckMentioned)

mentionChecker_system_prompt = """You are a medical reviewer assessing a narrative explanation of a medical test. Compare the list of positive findings or anomalities and list up the ones that are not mentioned in the narrative explanation.\n\nIf all the positive findings or anomalities are mentioned in the narrative explanation return an empty list."""
mentionChecker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", mentionChecker_system_prompt),
        ("human", "Positive findings and anomalities :\n{findings}\n\nNarrative explanation of test:\n{explanation}"),
    ]
)
mention_checker = mentionChecker_prompt | structured_mentionChecker



class explanationCheck(BaseModel):
    """Binary score for whether the narrative explanation about a test result is medically correct."""

    binary_score: str = Field(
        description="Whether the narrative explanation about a test result is medically correct, 'yes' or 'no'"
    )
structured_explanationChecker = llm4o.with_structured_output(explanationCheck)

explanationChecker_system_prompt = """You are a medical reviewer assessing a narrative explanation of a medical test. Compare the list of positive findings or anomalities and list up the ones that are not mentioned in the narrative explanation.\n\nIf all the positive findings or anomalities are mentioned in the narrative explanation return an empty list."""
explanationChecker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", explanationChecker_system_prompt),
        ("human", "Narrative explanation of test:\n{explanation}"),
    ]
)
explanation_checker = explanationChecker_prompt | structured_explanationChecker


class CommentType(BaseModel):
    """Class of the comment which is one of the following. 'f/u period'  and 'other'"""

    comment_class: str = Field(
        description="Class of comment. 'f/u period' or 'other'."
    )
structured_llm_comment_classifier = llm4omini.with_structured_output(CommentType)

commenttype_system_prompt = """ㅎiven a comment, classify the comment into two categories. 'f/u period' or 'other'. \n\n 'f/u period' means that the comment is only related with the follow-up period. 'other' means that the comment is not related to the follow-up period.\n\n"""
comment_type_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", commenttype_system_prompt),
        ("human", "Comment: {comment}"),
    ]
)
comment_classifier = comment_type_prompt | structured_llm_comment_classifier



class FixedExplanation(BaseModel):
    """A fixed explanation of the test report."""

    new_explanation: str = Field(
        description="Fixed explantation of the test report."
    )
structured_llm_explanation_fixer = llm4o.with_structured_output(FixedExplanation)

explanation_fixer_system_prompt = """Given a test report and a wrong explanation with comments explaning the error, edit the explanation in Korean so that it matches the test report. Try to keep a similiar tone and format"""
explanation_fixer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", explanation_fixer_system_prompt),
        ("human", "Test Report:\n{report}\n\nPrevious narrative explanation of the test: {explanation}\n\ncomment: {comment}"),
    ]
)
explanation_fixer = explanation_fixer_prompt | structured_llm_explanation_fixer





def biopsy(state):
    """
    Validates whether the a biopsy was performed.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    test_report = state["test_report"]
    
    
    response = Biopsy_checker.invoke({'report': test_report})
    biopsy = response.binary_score

    if biopsy == "no":
        return "limit examination check"
    else:
        return "end"


def limit_examination(state):
    """
    Validates whether the test had limited evaluation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    test_report = state["test_report"]
    explanation = state["explanation"]
    
    response = limit_checker.invoke({'report': test_report, 'explanation': explanation})
    
    if response.binary_score == "no":
        return {"comment": '제한된 검사라는 것이 올바르게 언급되지 않음      '}
    else : 
        return {"comment": ''}
        
        
def anomaly_list(state):
    """
    Gives a list of anomalies in the test report.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    test_report = state["test_report"]
    
    response = anomaly_checker.invoke({'report': test_report})
    
    list = response.AnomalyList
    return {"anomalies": list}

def anomaly_list_emptyornot(state):
    """
    Checks whether the list of anomalies is empty or not.
    
    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    
    a_list = state["anomalies"]
    
    if a_list == []:
        return "empty"
    else:
        return "not empty"
    
    


def nofindings_check(state):
    """
    Gives a list of anomalies in the narrative explanation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    explanation = state["explanation"]
    
    response = nofindings_checker.invoke({'explanation': explanation})
    
    list = response.AnomalyList
    list_string = ', '.join(list)
    
    if list == []:
        return {'comment': ''}
    else:
        return {'comment': f"결과에 언급되지 않은 {list_string}가 소견에 언급됨    "}


def mention_check(state):
    """
    Checks whether all the anomalies have been mentioned in the narrative explanation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    findings = state["anomalies"]
    explanation = state["explanation"]
    
    response = mention_checker.invoke({'findings': findings, 'explanation' : explanation})
    
    list = response.MissedFindings
    list_string = ', '.join(list)
    
    if list == []:
        return {'comment': ''}
    else:
        return {'comment': f"결과에서 언급된 {list_string}가 소견에서 언급되지 않음    "}
    
    
def explanation_check(state):
    """
    Checks whether all the anomalies have been mentioned in the narrative explanation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    explanation = state["explanation"]
    
    response = explanation_checker.invoke({'explanation' : explanation})
    
    score = response.binary_score
    
    if score == 'yes':
        return {'comment': ''}
    else:
        return {'comment': '소견에 의학적인 오류가 있음'}
    
def comment_type(state):
    """
    Classifies the comment of why the explanation doesn't match the test report.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    comment = state["comment"]
    
    response = comment_classifier.invoke({'comment':comment})
    comment_class = response.comment_class

    if comment_class == "other":
        return "generate"
    else:
        return "end"
    
def generate_new(state):
    """
    rewrites the explanation based on the comment and test report.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    explanation = state["explanation"]
    test_report = state["test_report"]
    comment = state["comment"]
    if comment == '':
        return {'comment' : ''}
    else :
        response = explanation_fixer.invoke({'explanation':explanation, 'report': test_report,'comment':comment})
        new_explanation = response.new_explanation
        return {"new_explanation": new_explanation}


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        test_report: test report
        explanation: original explanation of test report
        comment: comment of why the explanation doesn't match the test report
        new_explanation: new explanation of the test report
    """

    test_report: str
    explanation : str
    comment : Annotated[Optional[str],operator.add]
    new_explanation: Optional[str]
    anomalies : Optional[List[str]] 
    

workflow = StateGraph(GraphState)
 
# Define the nodes
workflow.add_node("limit_examination", limit_examination)  
workflow.add_node("anomaly_list", anomaly_list)  
workflow.add_node("nofindings_check", nofindings_check)
workflow.add_node("mention_check", mention_check)  
workflow.add_node("explanation_check", explanation_check)  
workflow.add_node("generate_new",generate_new)

# Build graph

workflow.add_conditional_edges(
    START,
    biopsy,
    {
        "limit examination check": "limit_examination",
        "end": END,
    },
)

workflow.add_edge("limit_examination", "anomaly_list")

workflow.add_conditional_edges(
    "anomaly_list",
    anomaly_list_emptyornot,
    {
        "empty": "nofindings_check",
        "not empty": "mention_check",
    },
)


workflow.add_edge("mention_check", "explanation_check")
workflow.add_conditional_edges(
    "explanation_check",
    comment_type,
    {
        "generate": "generate_new",
        "end": END
    },
)
workflow.add_conditional_edges(
    "nofindings_check",
    comment_type,
    {
        "generate": "generate_new",
        "end": END
    },
)
workflow.add_edge("generate_new", END)
# Compile
validation = workflow.compile()


    
