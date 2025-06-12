
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, TypedDict, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class Tasks(BaseModel):
    tasks: List[str] = Field(..., description='list of information to gather steps')

class SubQueries(BaseModel):
    questions: List[str] = Field(..., description='Queries to search using the web search tool')

class ReportFormat(BaseModel):
    content: List[str] = Field(..., description='Content written with propr citation fomatting ')
    citations: List[Tuple[int, str]] = Field(..., description='List of citations where each citation is a list containing the citation number and the weblink')

class SearchState(BaseModel):
    max_tasks: int = Field(default=5)
    max_subqueries: int = Field(default=3)
    max_query_results: int = Field(default=3)
    query: str
    current_task: Optional[str] = Field(default='')
    next_tasks: Optional[str] = Field(default='')
    current_subqueries: Optional[List[str]] = Field(default=[])
    findings: Optional[List[str]] = Field(default=[])
    messages: List[HumanMessage | AIMessage | SystemMessage]