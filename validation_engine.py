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
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT'] = st.secrets['LANGCHAIN_PROJECT']


llm4o = ChatOpenAI(model="gpt-4o", temperature=0)
llm4omini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
class GradeMatch(BaseModel):
    """Binary score for whether the explanation matches the test report."""

    binary_score: str = Field(
        description="Explanation matches the test report , 'yes' or 'no'"
    )
structured_llm_grader = llm4o.with_structured_output(GradeMatch)

hallucination_system_prompt = """You are a grader assessing whether an explantion is grounded and matches to a test report.\n\nGive a binary score 'yes' or 'no'. 'yes' means that the explanation matches the test report. Be sensitive about important positive findings and urgency mentioned. If you are not sure output 'no'"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_system_prompt),
        ("human", "Test Report: {report}\n\nTest Explanation: {explanation}"),
    ]
)
hallucination_grader = hallucination_prompt | structured_llm_grader

###comment of why it is mismatching
class MismatchComment(BaseModel):
    """Comment for why the explanation is mismatching the test report."""

    comment: str = Field(
        description=" A brief comment for why the explanation is mismatching the test report."
    )
structured_llm_commenter = llm4o.with_structured_output(MismatchComment)

comment_system_prompt = """When given a explanation that does not match the test report, give a comment why it doesn't match.\n\nBe as clear as possible."""
comment_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", comment_system_prompt),
        ("human", "Test Report: {report}\n\nTest Explanation: {explanation}"),
    ]
)
commenter = comment_prompt | structured_llm_commenter

class CommentType(BaseModel):
    """Class of the comment which is one of the following. 'f/u period', 'biopsy results','both' and 'other'"""

    comment_class: str = Field(
        description="Class of comment. 'f/u period', 'biopsy results', 'both' or 'other'."
    )
structured_llm_comment_classifier = llm4omini.with_structured_output(CommentType)

commenttype_system_prompt = """When given a comment classify the comment into three categories. 'f/u period', 'biopsy results', or 'other'. \n\n 'f/u period' means that the comment is only related to the follow-up period.\n\n'biopsy results' means that the comment is only related to the biopsy results.\n\n'both' means that the comment is related to to biopsy results and follow-up period only. \n\n 'other' means that the comment is not related to the follow-up period or biopsy results.\n\n"""
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

explanation_fixer_system_prompt = """Test report and an prewritten explanation about the test report with a comment, rewrite the explanation based on the comment and test report. Though try to keep a similiar tone and format"""
explanation_fixer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", explanation_fixer_system_prompt),
        ("human", "Test Report: {report}\n\nPrevious Explanation: {explanation}\n\nComment: {Comment}"),
    ]
)
explanation_fixer = explanation_fixer_prompt | structured_llm_explanation_fixer





def validate(state):
    """
    Validates whether the given explanation matches the test report.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    
    explanation = state["explantion"]
    test_report = state["test_report"]
    
    
    response = hallucination_grader.invoke({'explantion':explanation, 'report': test_report})
    validation = response.binary_score

    if validation == "no":
        return "comment"
    else:
        return "end"
    
    
def comment_node(state):
    """
    generates comment of why the explanation doesn't match the test report.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, match
    """
    explanation = state["explanation"]
    test_report = state["test_report"]
    
    response = commenter.invoke({'explantion':explanation, 'report': test_report})
    comment = response.comment
    return {"comment": comment}


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
    comment : Optional[str]
    new_explanation: Optional[str]
    

workflow = StateGraph(GraphState)
 
# Define the nodes
workflow.add_node("comment_node", comment_node)  
workflow.add_node("generate_new",generate_new)
# Build graph

workflow.add_conditional_edges(
    START,
    validate,
    {
        "comment": "comment_node",
        "end": END,
    },
)
workflow.add_conditional_edges(
    "comment_node",
    comment_type,
    {
        "generate": "generate_new",
        "end": END
    },
)
workflow.add_edge("generate_new", END)
# Compile
validation = workflow.compile()


    