
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class Tasks(BaseModel):
    tasks: List[str] = Field(..., description='list of information to gather steps')

class SubQueries(BaseModel):
    questions: List[str] = Field(..., description='Queries to search using the web search tool')

class WebLinkRelevancyVerdict(BaseModel):
    relevant: bool= Field(
        ..., 
        description='Whether the weblink is might possibly have relevant information to help answer the Sub Query')

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