
import ast
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import Any, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tavily import TavilyClient

from .data_structures import SearchState, SubQueries, Tasks
from .models import Models
from .prompts import (
    PlannerPrompts,
    ReportWriterPrompts,
    ReplannerPrompts,
    SubQueriesPrompts,
    WebSearchPrompts,
    fill_prompt,
)

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


EXCLUDE_URLS = [
    "https://www.tiktok.com/",
    "https://www.youtube.com/",
    "https://github.com/",
]


class QueryPlannerNode():
    """Plans high-level tasks from a user query.
    
    This node takes a user's search query and breaks it down into a series of
    actionable tasks that can be executed sequentially to comprehensively
    address the query.
    
    Attributes:
        llm: Language model configured with structured output for Tasks.
    """

    def __init__(self, llm=Models.gpt_4o_mini):
        """Initialize the QueryPlannerNode.
        
        Args:
            llm: Language model instance to use for planning. Defaults to
                Models.gpt_4o_mini configured with structured output.
        """
        self.llm = llm.with_structured_output(Tasks)
        logger.info("QueryPlannerNode initialized with %s", llm)

    def plan_query(
        self,
        max_tasks: int,
        history: List[Union[HumanMessage, AIMessage, SystemMessage]],
    ) -> Tasks:
        """Generate a structured plan of tasks from conversation history.
        
        Args:
            max_tasks: Maximum number of tasks to generate.
            history: List of messages representing the conversation history.
            
        Returns:
            Tasks: Structured object containing the list of planned tasks.
            
        Raises:
            Exception: If LLM invocation fails during planning.
        """
        logger.debug("Planning query: max_tasks=%d, history length=%d", max_tasks, len(history))

        system_prompt = fill_prompt(
            PlannerPrompts.system_prompt_format, max_tasks=max_tasks
        )
        messages = [SystemMessage(content=system_prompt)] + history

        tasks = self.llm.invoke(messages)
        logger.info("Generated %d tasks", len(tasks.tasks))
        return tasks

    def __call__(self, state: SearchState) -> SearchState:
        """Execute the query planning process.
        
        This method processes the current search state, generates tasks based on
        the user's query, and updates the state with the current task and
        remaining tasks.
        
        Args:
            state: Current search state containing the query and configuration.
            
        Returns:
            SearchState: Updated state with planned tasks and formatted messages.
        """
        logger.info("QueryPlannerNode called: max_tasks=%d", state.max_tasks)

        # Prepare user prompt
        raw_query = state.messages[-1].content
        user_prompt = fill_prompt(
            PlannerPrompts.user_prompt_format, query=raw_query
        )
        state.messages[-1] = HumanMessage(content=user_prompt)

        # Plan tasks
        tasks_obj = self.plan_query(state.max_tasks, state.messages)
        tasks = tasks_obj.tasks or []
        state.max_tasks -= 1

        # Extract current and next tasks
        current_task = tasks[0] if tasks else ""
        next_tasks_list = tasks[1:] if len(tasks) > 1 else []
        next_tasks_text = "\n- ".join(next_tasks_list)

        state.current_task = current_task
        state.next_tasks = next_tasks_text

        # Append formatted messages
        state.messages.append(
            AIMessage(
                content=fill_prompt(
                    PlannerPrompts.current_task_format, task=current_task
                )
            )
        )
        state.messages.append(
            AIMessage(
                content=fill_prompt(
                    PlannerPrompts.next_tasks_format, next_tasks=next_tasks_text
                )
            )
        )

        logger.info("Query planning complete, next max_tasks=%d", state.max_tasks)
        return state


class SubqueryGeneratorNode():
    """Generates subqueries for a given task.
    
    This node takes a high-level task and decomposes it into specific subqueries
    that can be used for web searching. It also handles query cleaning to ensure
    search compatibility.
    
    Attributes:
        llm: Language model configured with structured output for SubQueries.
    """

    def __init__(self, llm=Models.gpt_4o_mini):
        """Initialize the SubqueryGeneratorNode.
        
        Args:
            llm: Language model instance to use for subquery generation.
                Defaults to Models.gpt_4o_mini configured with structured output.
        """
        self.llm = llm.with_structured_output(SubQueries)
        logger.info("SubqueryGeneratorNode initialized with %s", llm)

    def generate_subqueries(self, max_subqueries: int, task: str) -> SubQueries:
        """Generate subqueries for a specific task.
        
        Args:
            max_subqueries: Maximum number of subqueries to generate.
            task: The high-level task to decompose into subqueries.
            
        Returns:
            SubQueries: Structured object containing the generated subqueries.
            
        Raises:
            Exception: If LLM invocation fails during subquery generation.
        """
        logger.debug("Generating %d subqueries for task: %s", max_subqueries, task)

        system_prompt = fill_prompt(
            SubQueriesPrompts.system_prompt_format, max_subqueries=max_subqueries
        )
        user_prompt = fill_prompt(
            SubQueriesPrompts.user_prompt_format, task=task
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        return self.llm.invoke(messages)

    @staticmethod
    def clean_query(q: str) -> str:
        """Clean a query string for search compatibility.
        
        Removes special characters and extra whitespace to ensure the query
        is compatible with search engines.
        
        Args:
            q: Raw query string to clean.
            
        Returns:
            str: Cleaned query string with only alphanumeric characters and spaces.
        """
        return re.sub(r"[^A-Za-z0-9 ]+", " ", q).strip()

    def __call__(self, state: SearchState) -> SearchState:
        """Execute the subquery generation process.
        
        This method takes the current task from the search state, generates
        subqueries for it, cleans the queries, and updates the state.
        
        Args:
            state: Current search state containing the task to process.
            
        Returns:
            SearchState: Updated state with generated subqueries and formatted messages.
        """
        logger.info("SubqueryGeneratorNode called for task: %s", state.current_task)
        subq_obj = self.generate_subqueries(state.max_subqueries, state.current_task)
        raw_questions = subq_obj.questions or []
        cleaned = [self.clean_query(q) for q in raw_questions]

        state.current_subqueries = cleaned
        state.messages.append(
            AIMessage(
                content=fill_prompt(
                    SubQueriesPrompts.subqueries_format,
                    subqueries="\n- " + "\n- ".join(cleaned),
                )
            )
        )

        logger.info("Generated %d subqueries", len(cleaned))
        return state


class WebSearchNode():
    """Performs web searches for subqueries and compiles answers.
    
    This node executes web searches using the Tavily API, processes and
    consolidates the results, and generates comprehensive answers for tasks.
    It supports concurrent searching, result filtering, and content consolidation.
    
    Attributes:
        context_cap: Maximum character limit for content processing.
        executor: ThreadPoolExecutor for concurrent search operations.
        websearch_consolidation_llm: LLM for consolidating search results.
        create_answer_llm: LLM for generating final answers.
        topic: Search topic category (general, news, finance).
        depth: Search depth (basic, advanced).
        time_range: Time range for search results.
        days: Number of days for time-based searches.
        include_urls: List of URLs to include in searches.
        exclude_urls: List of URLs to exclude from searches.
        threshold: Minimum relevance score threshold for results.
    """

    def __init__(
        self,
        context_cap: int = 120_000,
        concurrency_max_workers: int = 25,
        websearch_consolidation_llm=Models.gpt_4o_mini,
        create_answer_llm=Models.gpt_4o_mini,
        topic: str = "general",
        depth: str = "advanced",
        time_range: Optional[str] = "year",
        days: int = 7,
        include_urls: Optional[List[str]] = None,
        exclude_urls: Optional[List[str]] = None,
        threshold: float = 0.3,
    ):
        """Initialize the WebSearchNode with configuration options.
        
        Args:
            context_cap: Maximum character limit for content processing.
                Defaults to 120,000.
            concurrency_max_workers: Maximum number of concurrent search workers.
                Defaults to 25.
            websearch_consolidation_llm: LLM for consolidating search results.
                Defaults to Models.gpt_4o_mini.
            create_answer_llm: LLM for generating final answers.
                Defaults to Models.gpt_4o_mini.
            topic: Search topic category. Must be one of "general", "news", "finance".
                Defaults to "general".
            depth: Search depth. Must be "basic" or "advanced". Defaults to "advanced".
            time_range: Time range for searches. Must be one of "day", "week", "month", "year".
                Defaults to "year".
            days: Number of days for time-based searches. Must be >= 1. Defaults to 7.
            include_urls: List of URLs to specifically include in searches.
                Defaults to None.
            exclude_urls: List of URLs to exclude from searches.
                Defaults to EXCLUDE_URLS constant.
            threshold: Minimum relevance score threshold (0-1). Defaults to 0.3.
        """
        self.context_cap = context_cap
        self.executor = ThreadPoolExecutor(max_workers=concurrency_max_workers)
        self.websearch_consolidation_llm = websearch_consolidation_llm
        self.create_answer_llm = create_answer_llm

        self.topic = topic if topic in ("general", "news", "finance") else "general"
        self.depth = depth if depth in ("basic", "advanced") else "basic"
        self.time_range = time_range if time_range in ("day", "week", "month", "year") else None
        self.days = days if days >= 1 else 7

        self.include_urls = include_urls or []
        self.exclude_urls = exclude_urls or EXCLUDE_URLS
        self.threshold = float(threshold) if 0 <= threshold <= 1 else 0.5

        logger.info(
            "WebSearchNode init: topic=%s, depth=%s, time_range=%s, days=%d, "
            "workers=%d",
            self.topic,
            self.depth,
            self.time_range,
            self.days,
            concurrency_max_workers,
        )

    @staticmethod
    def get_content(search_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract content results from a search response.
        
        Args:
            search_response: Dictionary containing search results from API.
            
        Returns:
            List[Dict[str, Any]]: List of result dictionaries, or empty list if
                no results found.
        """
        return search_response.get("results", [])

    def filter_by_score(self, search_response: Dict[str, Any]) -> Dict[str, Any]:
        """Filter search results by relevance score threshold.
        
        Args:
            search_response: Dictionary containing search results with scores.
            
        Returns:
            Dict[str, Any]: Filtered search response containing only results
                that meet the score threshold.
        """
        results = search_response.get("results", [])
        filtered = [r for r in results if r.get("score", 0) >= self.threshold]
        return {**search_response, "results": filtered}

    def search_function(self, max_results: int, subquery: str) -> Dict[str, Any]:
        """Perform a web search for a single subquery.
        
        Uses the Tavily API to search for information related to the subquery,
        applying configured filters and parameters.
        
        Args:
            max_results: Maximum number of results to return.
            subquery: The search query string.
            
        Returns:
            Dict[str, Any]: Search response containing results and metadata.
                Returns empty results dict if search fails.
            
        Raises:
            EnvironmentError: If TAVILY_API_KEY environment variable is not set.
        """
        logger.debug("Searching for %r with max_results=%d", subquery, max_results)
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.error("TAVILY_API_KEY not set")
            raise EnvironmentError("TAVILY_API_KEY environment variable not set")

        client = TavilyClient(api_key=api_key)
        try:
            resp = client.search(
                query=subquery,
                topic=self.topic,
                search_depth=self.depth,
                max_results=max_results,
                time_range=self.time_range,
                days=self.days,
                include_answer=True,
                include_raw_content="text",
                include_domains=self.include_urls,
                exclude_domains=self.exclude_urls,
            )
        except Exception as e:
            logger.error("Tavily search failed: %s", e, exc_info=True)
            return {"results": []}

        return self.filter_by_score(resp)

    def deep_get(
        self,
        data: Union[Dict[str, Any], List[Any]],
        keys: Sequence[str],
        default: Any = None,
    ) -> Any:
        """Safely extract nested values from complex data structures.
        
        Recursively traverses nested dictionaries and lists to extract values
        using a sequence of keys, with fallback to default values.
        
        Args:
            data: The data structure to traverse (dict or list).
            keys: Sequence of keys to traverse in order.
            default: Default value to return if key path is not found.
                Defaults to None.
            
        Returns:
            Any: The value at the specified key path, or the default value
                if the path is not found.
        """
        for key in keys:
            try:
                if isinstance(data, list):
                    data = [self.deep_get(item, [key], default) for item in data]
                else:
                    data = data[key]
            except (KeyError, IndexError, TypeError):
                logger.warning("deep_get missing key %r, using default", key)
                return default
        return data

    def extract_and_consolidate(
        self, task: str, subquery: str, result: Dict[str, Any]
    ) -> str:
        """Extract and consolidate content from a single search result.
        
        Takes a search result, extracts relevant fields, and uses an LLM to
        consolidate the content into a focused summary relevant to the task.
        
        Args:
            task: The high-level task being addressed.
            subquery: The specific subquery that generated this result.
            result: Dictionary containing search result data.
            
        Returns:
            str: Consolidated content formatted with subquery header.
        """
        title = self.deep_get(result, ["title"], "")
        url = self.deep_get(result, ["url"], "")
        snippet = self.deep_get(result, ["content"], "")
        raw = self.deep_get(result, ["raw_content"], snippet)
        content = f"Source: {title} ({url})\n{raw}"[: self.context_cap]

        # Consolidate via LLM
        prompt = fill_prompt(
            WebSearchPrompts.consolidation_user_prompt_format,
            task=task,
            subquery=subquery,
            content=content,
        )
        msgs = [
            SystemMessage(content=WebSearchPrompts.consolidation_system_prompt),
            HumanMessage(content=prompt),
        ]
        try:
            consolidated = self.websearch_consolidation_llm.invoke(msgs).content
        except Exception:
            consolidated = content
        return f"SUBQUERY: {subquery}\n{consolidated}"

    def group_results(
        self, subqueries: List[str], contents: List[str]
    ) -> tuple[List[str], List[str]]:
        """Group search results by subquery.
        
        Organizes search results by their corresponding subqueries and
        concatenates multiple results for the same subquery.
        
        Args:
            subqueries: List of subquery strings.
            contents: List of content strings corresponding to subqueries.
            
        Returns:
            tuple[List[str], List[str]]: A tuple containing:
                - List of unique subqueries
                - List of grouped content strings (one per subquery)
        """
        grouped = defaultdict(list)
        for sq, ct in zip(subqueries, contents):
            grouped[sq].append(ct)

        keys = list(grouped.keys())
        texts = [
            f"SUBQUERY: {k}\n" + "\n\n-----next article-----\n\n".join(grouped[k])[: self.context_cap]
            for k in keys
        ]
        return keys, texts

    def create_answer(self, task: str, content: str) -> str:
        """Generate a comprehensive answer for a task based on search content.
        
        Uses an LLM to synthesize search results into a coherent answer that
        addresses the specific task requirements.
        
        Args:
            task: The high-level task being addressed.
            content: Consolidated search content from multiple sources.
            
        Returns:
            str: Generated answer addressing the task, or the original content
                if answer generation fails.
        """
        prompt = fill_prompt(
            WebSearchPrompts.task_answer_user_prompt_format,
            task=task,
            content=content,
        )
        msgs = [
            SystemMessage(content=WebSearchPrompts.task_answer_system_prompt),
            HumanMessage(content=prompt),
        ]
        try:
            return self.create_answer_llm.invoke(msgs).content
        except Exception as e:
            logger.error("Answer creation failed: %s", e)
            return content

    def __call__(self, state: SearchState) -> SearchState:
        """Execute the web search and answer generation process.
        
        This method coordinates the entire web search workflow: searching for
        subqueries, extracting and consolidating results, grouping by subquery,
        and generating a final answer.
        
        Args:
            state: Current search state containing subqueries and task information.
            
        Returns:
            SearchState: Updated state with search findings and generated answer.
        """
        logger.info("WebSearchNode called: task=%r", state.current_task)

        # 1. Search
        responses = self.executor.map(
            self.search_function, repeat(state.max_query_results), state.current_subqueries
        )
        responses = list(responses)

        # 2. Extract + consolidate
        all_subq, all_contents = [], []
        for subq, resp in zip(state.current_subqueries, responses):
            for item in self.get_content(resp):
                all_subq.append(subq)
                all_contents.append(item)

        results = list(
            self.executor.map(
                self.extract_and_consolidate,
                repeat(state.current_task),
                all_subq,
                all_contents
            )
        )

        # 3. Group
        grouped_subq, grouped_texts = self.group_results(all_subq, results)

        # 4. Final answer
        answer = self.create_answer(state.current_task, "\n\n".join(grouped_texts))

        findings = fill_prompt(
            WebSearchPrompts.findings_format,
            task=state.current_task,
            content=answer,
        )
        state.findings.append(findings)
        state.messages.append(AIMessage(content=findings))

        logger.info("WebSearchNode complete for task=%r", state.current_task)
        return state


class ReplannerNode():
    """Replans tasks based on accumulated findings.
    
    This node analyzes the accumulated research findings and adjusts the
    remaining task plan to ensure comprehensive coverage of the original query.
    It can modify, add, or remove tasks based on what has been discovered.
    
    Attributes:
        llm: Language model configured with structured output for Tasks.
    """

    def __init__(self, llm=Models.gpt_4o_mini):
        """Initialize the ReplannerNode.
        
        Args:
            llm: Language model instance to use for replanning.
                Defaults to Models.gpt_4o_mini configured with structured output.
        """
        self.llm = llm.with_structured_output(Tasks)
        logger.info("ReplannerNode initialized with %s", llm)

    def replan(
        self, max_tasks: int, query: str, next_tasks: str, findings: str
    ) -> Tasks:
        """Generate a revised task plan based on current findings.
        
        Analyzes the accumulated findings and remaining tasks to create an
        updated plan that addresses any gaps or new directions discovered
        during research.
        
        Args:
            max_tasks: Maximum number of tasks to generate in the new plan.
            query: The original user query.
            next_tasks: String representation of currently planned next tasks.
            findings: Accumulated research findings from completed tasks.
            
        Returns:
            Tasks: Structured object containing the revised task list.
            
        Raises:
            Exception: If LLM invocation fails during replanning.
        """
        system_prompt = fill_prompt(
            ReplannerPrompts.system_prompt_format, max_tasks=max_tasks
        )
        user_prompt = fill_prompt(
            ReplannerPrompts.user_prompt_format,
            query=query,
            next_tasks=next_tasks,
            findings=findings,
        )
        msgs = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return self.llm.invoke(msgs)

    def __call__(self, state: SearchState) -> SearchState:
        """Execute the task replanning process.
        
        This method analyzes the current state, including all accumulated
        findings, and generates a revised task plan. Updates the state with
        the new current task and remaining tasks.
        
        Args:
            state: Current search state with findings and task information.
            
        Returns:
            SearchState: Updated state with revised task plan and formatted messages.
        """
        logger.info("ReplannerNode called: max_tasks=%d", state.max_tasks)
        findings_text = "\n-----\n".join(state.findings)
        tasks_obj = self.replan(
            state.max_tasks, state.query, state.next_tasks, findings_text
        )
        tasks = tasks_obj.tasks or []
        state.max_tasks -= 1

        state.current_task = tasks[0] if tasks else ""
        state.next_tasks = "\n- ".join(tasks[1:]) if len(tasks) > 1 else ""

        state.messages.append(
            AIMessage(
                content=fill_prompt(
                    PlannerPrompts.current_task_format, task=state.current_task
                )
            )
        )
        state.messages.append(
            AIMessage(
                content=fill_prompt(
                    PlannerPrompts.next_tasks_format, next_tasks=state.next_tasks
                )
            )
        )

        logger.info("ReplannerNode complete, new max_tasks=%d", state.max_tasks)
        return state


class ReportWriterNode():
    """Generates final report from all findings.
    
    This node synthesizes all accumulated research findings into a comprehensive
    final report that addresses the original user query. It structures the
    information logically and presents it in a clear, readable format.
    
    Attributes:
        llm: Language model instance used for report generation.
    """

    def __init__(self, llm=Models.gpt_4o_mini):
        """Initialize the ReportWriterNode.
        
        Args:
            llm: Language model instance to use for report writing.
                Defaults to Models.gpt_4o_mini.
        """
        self.llm =llm
        logger.info("ReportWriterNode initialized with %s", llm)

    def write_report(self, query: str, findings: str) -> str:
        """Generate a comprehensive report from research findings.
        
        Takes all accumulated research findings and synthesizes them into
        a well-structured report that comprehensively addresses the original
        user query.
        
        Args:
            query: The original user query that initiated the research.
            findings: All accumulated research findings from completed tasks.
            
        Returns:
            str: Generated comprehensive report, or the raw findings if
                report generation fails.
        """
        prompt = fill_prompt(
            ReportWriterPrompts.user_prompt_format, query=query, findings=findings
        )
        msgs = [
            SystemMessage(content=ReportWriterPrompts.system_prompt),
            HumanMessage(content=prompt),
        ]
        try:
            return self.llm.invoke(msgs).content
        except Exception as e:
            logger.error("Report generation failed: %s", e)
            return findings

    def __call__(self, state: SearchState) -> SearchState:
        """Execute the report generation process.
        
        This method takes all accumulated findings from the search state and
        generates a comprehensive final report. The report is added to the
        message history as the final response.
        
        Args:
            state: Current search state containing all accumulated findings.
            
        Returns:
            SearchState: Updated state with the final report added to messages.
        """
        logger.info("ReportWriterNode called for query=%r", state.query)
        findings_text = "\n-----\n".join(state.findings)
        report = self.write_report(state.query, findings_text)
        state.messages.append(AIMessage(content=report))
        logger.info("ReportWriterNode complete")
        return state

# import os
# import re
# import logging
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor
# from itertools import repeat
# from langchain_core.messages import (
#     HumanMessage, 
#     AIMessage, 
#     SystemMessage
# )

# from typing import (
#     List,
#     Literal,
#     Union,
#     Any
# )
# from tavily import TavilyClient

# from dotenv import load_dotenv

# load_dotenv()

# from .models import Models
# from .prompts import (
#     PlannerPrompts,
#     SubQueriesPrompts,
#     WebSearchPrompts,
#     ReplannerPrompts,
#     ReportWriterPrompts,
#     fill_prompt
# )
# from .data_structures import (
#     Tasks,
#     SubQueries,
#     WebLinkRelevancyVerdict,
#     SearchState
# )

# # Configure logger
# logger = logging.getLogger(__name__)

# # URLs to exclude from search results
# EXCLUDE_URLS = [
#     "https://www.tiktok.com/",
#     "https://www.youtube.com/",
#     "https://github.com/",
# ]


# class QueryPlannerNode:
#     """
#     A node responsible for planning queries by determining tasks to be performed.
    
#     This class takes a user query and generates a structured plan of tasks to address the query.
#     It uses an LLM to analyze the search context and break down complex queries into manageable tasks.
    
#     Attributes:
#         llm: Language model configured to output structured Tasks objects
#     """

#     def __init__(self, llm=Models.gpt_4o_mini):
#         """
#         Initialize the QueryPlannerNode.
        
#         Args:
#             llm: The language model to use for task planning (defaults to gpt-4o-mini)
#         """
#         self.llm = llm.with_structured_output(Tasks)
#         logger.info("QueryPlannerNode initialized with %s model", llm)

#     def plan_query(
#         self,
#         max_tasks: int, 
#         history: List[HumanMessage | AIMessage | SystemMessage]
#     ) -> Tasks:
#         """
#         Plan a search query by generating a list of tasks.
        
#         Args:
#             max_tasks: The current task number in the search process
#             history: The conversation history
            
#         Returns:
#             Tasks: Structured object containing the list of tasks to be performed
#         """
#         logger.debug("Planning query for task %d with history length %d", max_tasks, len(history))
        
#         system_prompt = fill_prompt(
#             PlannerPrompts.system_prompt_format,
#             max_tasks=max_tasks
#         )

#         messages = [
#             SystemMessage(content=system_prompt)
#         ] + history

#         tasks = self.llm.invoke(messages)
#         logger.info("Generated %d tasks for query planning", len(tasks.tasks))
        
#         return tasks

#     def __call__(
#         self,
#         state: SearchState
#     ) -> SearchState:
#         """
#         Process the current state to plan next tasks.
        
#         Args:
#             state: The current search state
            
#         Returns:
#             SearchState: Updated search state with planned tasks
#         """
#         logger.info("QueryPlannerNode called with max_tasks=%d", state.max_tasks)
        
#         query = state.messages[-1]
#         query = fill_prompt(
#             PlannerPrompts.user_prompt_format,
#             query=query.content
#         )

#         state.messages[-1] = HumanMessage(content=query)

#         tasks = self.plan_query(
#             state.max_tasks,
#             state.messages
#         ).tasks
        
#         current_task = ""
#         next_tasks = ""

#         if tasks:
#             current_task = tasks[0]
#             logger.debug("Current task: %s", current_task)

#         if len(tasks) > 1:
#             next_tasks = '- ' + "\n- ".join(tasks[1:])
#             logger.debug("Next tasks defined: %d tasks", len(tasks) - 1)

#         state.current_task = current_task
#         state.next_tasks = next_tasks

#         next_tasks_msg = fill_prompt(
#             PlannerPrompts.next_tasks_format,
#             next_tasks=next_tasks
#         )
#         state.messages += [AIMessage(content=next_tasks_msg)]

#         current_task_msg = fill_prompt(
#             PlannerPrompts.current_task_format,
#             task=current_task
#         )


#         state.messages += [AIMessage(content=current_task)]
#         state.max_tasks -= 1
        
#         logger.info("Query planning complete, max_tasks decremented to %d", state.max_tasks)
#         return state

    
# class SubqueryGeneratorNode:
#     """
#     A node responsible for generating subqueries for a given task.
    
#     This class takes a high-level task and breaks it down into specific subqueries
#     that can be used for web searches to gather relevant information.
    
#     Attributes:
#         llm: Language model configured to output structured SubQueries objects
#     """
    
#     def __init__(
#         self, 
#         llm=Models.gpt_4o_mini
#     ):
#         """
#         Initialize the SubqueryGeneratorNode.
        
#         Args:
#             llm: The language model to use for subquery generation (defaults to gpt-4o-mini)
#         """
#         self.llm = llm.with_structured_output(SubQueries)
#         logger.info("SubqueryGeneratorNode initialized with %s model", llm)

#     def generate_subquery(
#         self,
#         max_subqueries: int, 
#         task: str
#     ) -> SubQueries:
#         """
#         Generate subqueries for a given task.
        
#         Args:
#             max_subqueries: The maximum number of subqueries to generate
#             task: The task for which to generate subqueries
            
#         Returns:
#             SubQueries: Structured object containing the list of generated subqueries
#         """
#         logger.debug("Generating up to %d subqueries for task: %s", max_subqueries, task)
        
#         system_prompt = fill_prompt(
#             SubQueriesPrompts.system_prompt_format,
#             max_subqueries=max_subqueries
#         )

#         user_prompt = fill_prompt(
#             SubQueriesPrompts.user_prompt_format,
#             task=task
#         )

#         messages = [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=user_prompt),
#         ]

#         subqueries = self.llm.invoke(messages)
#         logger.info("Generated %d subqueries", len(subqueries.questions))
        
#         return subqueries
    
#     @staticmethod
#     def clean_query(s: str) -> str:
#         """
#         Clean a query string by removing non-alphanumeric characters.
        
#         Args:
#             s: The query string to clean
            
#         Returns:
#             str: The cleaned query string
#         """
#         return re.sub(r'[^a-zA-Z0-9 ]', ' ', s)

#     def __call__(
#         self,
#         state: SearchState
#     ) -> SearchState:
#         """
#         Process the current state to generate subqueries for the current task.
        
#         Args:
#             state: The current search state
            
#         Returns:
#             SearchState: Updated search state with generated subqueries
#         """
#         logger.info("SubqueryGeneratorNode called for task: %s", state.current_task)
        
#         questions = self.generate_subquery(
#             state.max_subqueries,
#             state.current_task
#         ).questions

#         questions = [self.clean_query(q) for q in questions]
#         logger.debug("Cleaned subqueries: %s", questions)

#         response = fill_prompt(
#             SubQueriesPrompts.subqueries_format,
#             subqueries="\n- ".join(questions)
#         )

#         state.messages += [AIMessage(content=response)]
#         state.current_subqueries = questions
        
#         logger.info("Subquery generation complete, generated %d subqueries", len(questions))
#         return state


# class WebSearchNode:
#     """
#     A node responsible for performing web searches based on subqueries and processing the results.
    
#     This class handles the entire web search workflow, including searching for information,
#     evaluating the relevance of search results, extracting content from web links,
#     consolidating information, and creating a comprehensive answer for the given task.
    
#     Attributes:
#         context_cap: Maximum character length for context
#         weblink_evaluator_llm: Language model for evaluating the relevance of web links
#         websearch_consolidation_llm: Language model for consolidating search results
#         create_answer_llm: Language model for creating the final answer
#         topic: Search topic focus
#         depth: Search depth level
#         time_range: Time range for search results
#         exclude_urls: List of URLs to exclude from search results
#         executor: ThreadPoolExecutor for concurrent processing
#     """
    
#     def __init__(
#         self,
#         context_cap: int = 120000,
#         concurrency_max_workers: int = 25,
#         weblink_evaluator_llm=Models.gpt_4o_mini,
#         websearch_consolidation_llm=Models.gpt_4o_mini,
#         create_answer_llm=Models.gpt_4o_mini,
#         topic: Literal['general', 'news', 'finance'] = 'general',
#         depth: Literal['basic', 'advanced'] = 'advanced',
#         time_range: Literal['day', 'month', 'week', 'year', 'month', None] = 'year',
#         days: int = 7,
#         include_urls: List[str] = [],
#         exclude_urls: List[str] = EXCLUDE_URLS,
#         threshold: float = 0.3
#     ):
#         """
#         Initialize the WebSearchNode.
        
#         Args:
#             context_cap: Maximum character length for context
#             concurrency_max_workers: Maximum number of concurrent workers for thread pool
#             weblink_evaluator_llm: Language model for evaluating web links
#             websearch_consolidation_llm: Language model for consolidating search results
#             create_answer_llm: Language model for creating the final answer
#             topic: Search topic focus (general, news, finance)
#             depth: Search depth level (basic, advanced)
#             time_range: Time range for search results (day, week, month, year)
#             exclude_urls: List of URLs to exclude from search results
#         """
#         self.context_cap = context_cap
#         self.weblink_evaluator_llm = weblink_evaluator_llm.with_structured_output(WebLinkRelevancyVerdict)
#         self.websearch_consolidation_llm = websearch_consolidation_llm
#         self.topic = topic
#         self.depth = depth
#         self.time_range = time_range
#         self.days = days
#         self.create_answer_llm = create_answer_llm
#         self.include_urls = include_urls
#         self.exclude_urls = exclude_urls
#         self.threshold = threshold

#         self.process_search_settings()

#         self.executor = ThreadPoolExecutor(
#             max_workers=concurrency_max_workers
#         )
        
#         logger.info(
#             "WebSearchNode initialized with %d workers, topic=%s, depth=%s, time_range=%s",
#             concurrency_max_workers, topic, depth, time_range
#         )


#     @staticmethod
#     def get_content(search_response: dict) -> List:
#         """
#         Extract content from a search response.
        
#         Args:
#             search_response: The search response dictionary
            
#         Returns:
#             List: The extracted content
#         """
#         content = []
#         try:
#             content = search_response['results']
#         except Exception as e:
#             logger.error("Error: No key 'results' in search response: %s", e)
#         return content

#     def process_search_settings(self):
#         # Validate depth
#         if self.depth not in ("basic", "advanced"):
#             logger.warning(f"Invalid depth '%s' - defaulted to 'basic'.", self.depth)
#             self.depth = "basic"

#         # Validate topic
#         if self.topic not in ("general", "news"):
#             logger.warning(f"Invalid topic '%s' - defaulted to 'general'.", self.topic)
#             self.topic = "general"

#         # Validate time_range
#         valid_ranges = {"day", "week", "month", "year", "d", "w", "m", "y"}
#         if self.time_range not in valid_ranges:
#             logger.warning(f"Invalid time_range '%s' - defaulted to None.", self.time_range)
#             self.time_range = None

#         # Validate days
#         if self.days < 1:
#             logger.warning(f"Invalid days '%s' - defaulted to 7.", self.days)
#             self.days = 7

#         # Parse include_domains
#         try:
#             self.include_domains = (
#                 ast.literal_eval(self.include_domains)
#                 if isinstance(self.include_domains, str)
#                 else self.include_domains
#             )
#             if isinstance(self.include_domains, list):
#                 self.include_domains = [item for item in self.include_domains if isinstance(item, str)]
#             else:
#                 raise ValueError()
#         except Exception:
#             logger.warning("Failed to parse include_domains - defaulted to empty list.")
#             self.include_domains = []

#         # Parse exclude_domains
#         try:
#             self.exclude_domains = (
#                 ast.literal_eval(self.exclude_domains)
#                 if isinstance(self.exclude_domains, str)
#                 else self.exclude_domains
#             )
#             if isinstance(self.exclude_domains, list):
#                 self.exclude_domains = [item for item in self.exclude_domains if isinstance(item, str)]
#             else:
#                 raise ValueError()
#         except Exception:
#             logger.warning("Failed to parse exclude_domains - defaulted to empty list.")
#             self.exclude_domains = []

#         # Parse search threshold
#         if not (isinstance(self.threshold, (float, int)) and 0 <= float(self.threshold) <= 1):
#             logger.warning(f"Invalid threshold '%s' - defaulted to 0.5.", self.threshold)
#             self.threshold = 0.5
#         else:
#             self.threshold = float(self.threshold)

#     def deep_get(
#         self,
#         _dictionary: Union[dict, List],
#         keys: str, 
#         default: Any = None
#     ) -> Any:
#         """
#         Safely get a nested value from a dictionary or list using a sequence of keys.
        
#         Args:
#             _dictionary: The dictionary or list to get values from
#             keys: Sequence of keys to navigate through the nested structure
#             default: Default value to return if keys don't exist
            
#         Returns:
#             Any: The value at the specified path, or the default if not found
#         """
#         for key in keys:
#             try:
#                 if isinstance(_dictionary, list):
#                     _dictionary = [self.deep_get(item, [key], default) for item in _dictionary]
#                 else:
#                     _dictionary = _dictionary[key]
#             except (KeyError, IndexError, TypeError):
#                 logger.warning(f"Deep_get Invalid key {key} returning {default}")
#                 return default
#         return _dictionary

#     def filter_by_score(self, search_results):
#         if not search_results or "results" not in search_results:
#             return search_results

#         filtered_results = [
#             result for result in search_results["results"]
#             if result.get("score", 0) >= self.threshold
#         ]

#         # Return a copy with filtered results
#         return {**search_results, "results": filtered_results}

#     def search_function(
#         self,
#         max_results: int,
#         subquery: str,
#     ) -> dict:
#         """
#         Perform a web search for a given subquery.
        
#         Args:
#             max_results: Maximum number of results to return
#             subquery: The search query
            
#         Returns:
#             dict: The search results
#         """
#         logger.debug("Searching for subquery: %s with max_results=%d", subquery, max_results)

#         api_key = os.environ.get("TAVILY_API_KEY")
#         if not api_key:
#             logger.error("TAVILY_API_KEY environment variable not set")
#             raise ValueError("TAVILY_API_KEY environment variable not set.")

#         tavily_client = TavilyClient(api_key=api_key)

#         response = tavily_client.search(
#             query=subquery,
#             topic=self.topic,
#             search_depth=self.depth,
#             max_results=max_results,
#             time_range=self.time_range,
#             days=self.days,
#             include_answer=True,
#             include_raw_content='text',
#             include_domains=self.include_urls,
#             exclude_domains=self.exclude_urls
#         )

#         return self.filter_by_score(response)

#     def extract_linked_content_n_format(
#         self,
#         current_task,
#         current_subquery,
#         search_result: dict
#     ) -> str:
#         """
#         Extract and format content from a search result.
        
#         Args:
#             search_result: The search result dictionary
            
#         Returns:
#             str: The formatted content
#         """
#         title = self.deep_get(
#             _dictionary=search_result,
#             keys=['title'],
#             default=''
#         )

#         link = self.deep_get(
#             _dictionary=search_result,
#             keys=['url'],
#             default=''
#         )
        

#         snippet = self.deep_get(
#             _dictionary=search_result,
#             keys=['content'],
#             default=''
#         )

#         content = self.deep_get(
#             _dictionary=search_result,
#             keys=['raw_content'],
#             default=snippet
#         )

#         content = f"Source: {title}({link})\n{content}"[:self.context_cap]

#         content = self.websearch_consolidation(
#             current_task,
#             current_subquery, 
#             content
#         )
#         return content

#     def group_subqueries_n_results(
#         self,
#         subqueries: List[str],
#         results: List[str]
#     ) -> tuple[List[str], List[str]]:
#         """
#         Group search results by their corresponding subqueries.
        
#         Args:
#             subqueries: List of subqueries
#             results: List of search results
            
#         Returns:
#             tuple: Grouped subqueries and their corresponding results
#         """
#         logger.debug("Grouping %d results for %d subqueries", len(results), len(set(subqueries)))
        
#         grouped_answers = defaultdict(list)
#         for subq, res in zip(subqueries, results):
#             grouped_answers[subq].append(res)
        
#         text_group_answers = []
#         grouped_subqueries = list(grouped_answers.keys())
#         for s in grouped_subqueries:
#             text_group_answers.append(
#                 f"SUBQUERY: {s}" + "\n\n----------next article----------\n\n".join(
#                     grouped_answers[s]
#                 )[:self.context_cap] 
#             )
        
#         logger.info("Grouped into %d unique subqueries", len(grouped_subqueries))
#         return grouped_subqueries, text_group_answers

#     def websearch_consolidation(
#         self,
#         task: str, 
#         subquery: str, 
#         content: str
#     ) -> str:
#         """
#         Consolidate search results for a subquery.
        
#         Args:
#             task: The main task
#             subquery: The subquery
#             content: The content to consolidate
            
#         Returns:
#             str: The consolidated content
#         """
#         logger.debug("Consolidating search results for subquery: %s", subquery)
        
#         user_prompt = fill_prompt(
#             WebSearchPrompts.consolidation_user_prompt_format,
#             task=task,
#             subquery=subquery,
#             content=content
#         )

#         messages = [
#             SystemMessage(content=WebSearchPrompts.consolidation_system_prompt),
#             HumanMessage(content=user_prompt),
#         ]

#         answer = self.websearch_consolidation_llm.invoke(messages).content
#         answer = f"SUBQUERY: {subquery}\n{content}"
        
#         logger.debug("Consolidation complete for subquery: %s", subquery)
#         return answer

#     def create_answer_for_task(
#         self,
#         task: str,
#         content: str
#     ) -> str:
#         """
#         Create a comprehensive answer for the main task based on consolidated content.
        
#         Args:
#             task: The main task
#             content: The consolidated content
            
#         Returns:
#             str: The comprehensive answer
#         """
#         logger.info("Creating final answer for task: %s", task)
        
#         user_prompt = fill_prompt(
#             WebSearchPrompts.task_answer_user_prompt_format,
#             task=task,
#             content=content
#         )

#         messages = [
#             SystemMessage(content=WebSearchPrompts.task_answer_system_prompt),
#             HumanMessage(content=user_prompt),
#         ]

#         answer = self.create_answer_llm.invoke(messages).content
#         logger.info("Answer created for task: %s", task)
#         return answer

#     def __call__(
#         self,
#         state: SearchState
#     ) -> SearchState:
#         """
#         Process the current state to perform web searches and create an answer.
        
#         Args:
#             state: The current search state
            
#         Returns:
#             SearchState: Updated search state with search results and answer
#         """
#         logger.info("WebSearchNode called for task: %s with %d subqueries", 
#                    state.current_task, len(state.current_subqueries))
        
#         current_task = state.current_task
#         current_subqueries = state.current_subqueries

#         # Perform searches for each subquery
#         logger.debug("Executing %d searches concurrently", len(current_subqueries))
#         search_responses = list(
#             self.executor.map(
#                 self.search_function,
#                 repeat(state.max_query_results),
#                 current_subqueries
#             )
#         )

#         # Extract search results
#         subqueries_w_results = []
#         results = []
#         for subquery, search_response in zip(current_subqueries, search_responses):
#             extra_content = self.get_content(search_response)
#             for c in extra_content:
#                 subqueries_w_results.append(subquery)
#                 results.append(c)
        
#         logger.info("Extracted %d total search results", len(results))

#         # Log detailed information if in debug mode
#         for i, j in zip(subqueries_w_results, results):
#             logger.debug("Subquery: %s", i)
#             logger.debug("Result: %s", j)
        

#         # Extract and format content from links
#         logger.info("Extracting raw content from %d relevant links", len(results))
#         results = list(
#             self.executor.map(
#                 self.extract_linked_content_n_format,
#                 repeat(current_task),
#                 subqueries_w_results,
#                 results
#             )
#         )
        
#         # Group results by subquery
#         subqueries_w_results, results = self.group_subqueries_n_results(
#             subqueries_w_results, 
#             results
#         )

#         # Log detailed information if in debug mode
#         for i, j in zip(subqueries_w_results, results):
#             logger.debug("Grouped subquery: %s", i)
#             logger.debug("Grouped results length: %d", len(j))

#         # Create comprehensive answer
#         logger.info("Creating comprehensive answer for task")
#         answer = self.create_answer_for_task(current_task, "\n\n".join(results))

#         # Format findings
#         findings = fill_prompt(
#             WebSearchPrompts.findings_format,
#             task=task,
#             content=answer
#         )

#         # Update state
#         state.findings += [findings]
#         state.messages += [AIMessage(content=findings)]
        
#         logger.info("WebSearchNode processing complete, answer added to findings")
#         return state

# class ReplannerNode:
#     """
#     A node responsible for dynamically replanning the search query execution based on 
#     intermediate findings and remaining tasks.
    
#     The ReplannerNode adapts the query execution plan as the search progresses by analyzing
#     current findings and determining what tasks should be executed next. It helps optimize 
#     the search process by prioritizing tasks based on accumulated information.
    
#     Attributes:
#         llm: The language model used for generating replanning decisions, configured to 
#              output structured Tasks objects.
#     """
#     def __init__(self, llm=Models.gpt_4o_mini):
#         """
#         Initialize a ReplannerNode with a specified language model.
        
#         Args:
#             llm: The language model to use for replanning decisions. 
#                  Defaults to gpt_4o_mini.
#         """
#         self.llm = llm.with_structured_output(Tasks)
#         logger.info("ReplannerNode initialized with %s model", llm)

#     def replan_query(
#         self,
#         max_tasks: int, 
#         query: str,
#         next_tasks: str,
#         findings: str
#     ) -> Tasks:
#         """
#         Replan the query execution based on current findings and remaining tasks.
        
#         This method generates a new task plan based on the original query, current findings,
#         and the remaining tasks. It sends these inputs to the language model to get an
#         updated set of tasks prioritized for execution.
        
#         Args:
#             max_tasks: The current task number in the search process.
#             query: The original search query.
#             next_tasks: String representation of the remaining tasks.
#             findings: Current accumulated findings from previous tasks.
            
#         Returns:
#             Tasks: A structured object containing the reprioritized tasks.
#         """
#         logger.debug("Replanning query at task %d with %d characters of findings", 
#                     max_tasks, len(findings))
        
#         system_prompt = fill_prompt(
#             ReplannerPrompts.system_prompt_format,
#             max_tasks=max_tasks
#         )

#         user_prompt = fill_prompt(
#             ReplannerPrompts.user_prompt_format,
#             query=query,
#             next_tasks=next_tasks,
#             findings=findings
#         )

#         messages = [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=user_prompt)
#         ]

#         logger.debug("Sending replanning request to language model")
#         tasks = self.llm.invoke(messages)
#         logger.info("Replanning completed, generated %d tasks", len(tasks.tasks) if hasattr(tasks, 'tasks') else 0)
        
#         return tasks

#     def __call__(
#         self,
#         state: SearchState
#     ) -> SearchState:
#         """
#         Process the current search state and update it with replanned tasks.
        
#         This method serves as the main entry point for the ReplannerNode. It takes the
#         current search state, generates a new plan, and updates the state with the
#         reprioritized tasks.
        
#         Args:
#             state: The current SearchState containing query, findings, and task information.
            
#         Returns:
#             SearchState: The updated state with replanned tasks and messages.
#         """
#         logger.info("Processing search state at task %d", state.max_tasks)

#         consolidated_findings = "-----\n\n".join(state.findings)
#         logger.debug("Consolidated %d findings (%d total characters) for report generation", 
#                     len(state.findings), len(consolidated_findings))
        
#         tasks = self.replan_query(
#             state.max_tasks,
#             state.query,
#             state.next_tasks,
#             consolidated_findings
#         ).tasks
        
#         current_task = ""
#         next_tasks = ""

#         if tasks:
#             current_task = tasks[0]
#             logger.info("Selected new current task: %s", current_task[:50] + "..." if len(current_task) > 50 else current_task)

#         if len(tasks) > 1:
#             next_tasks = '- ' + "\n- ".join(tasks[1:])
#             logger.debug("Queued %d additional tasks", len(tasks) - 1)

#         state.current_task = current_task
#         state.next_tasks = next_tasks

#         next_tasks_msg = fill_prompt(
#             PlannerPrompts.next_tasks_format,
#             next_tasks=next_tasks
#         )
#         state.messages += [AIMessage(content=next_tasks_msg)]

#         current_task_msg = fill_prompt(
#             PlannerPrompts.current_task_format,
#             task=current_task
#         )
#         state.messages += [AIMessage(content=current_task_msg)]
#         state.max_tasks -= 1
        
#         logger.debug("Search state updated, moving to task %d", state.max_tasks)
#         return state

# class ReportWriterNode:
#     """
#     A node responsible for generating comprehensive reports based on search queries and 
#     accumulated findings.
    
#     The ReportWriterNode transforms raw search findings into structured, coherent reports
#     that directly address the original query. It leverages a language model to synthesize
#     information and present it in a meaningful format.
    
#     Attributes:
#         llm: The language model used for report generation, configured to output 
#              structured Tasks objects.
#     """
#     def __init__(self, llm=Models.gpt_4o_mini):
#         """
#         Initialize a ReportWriterNode with a specified language model.
        
#         Args:
#             llm: The language model to use for report generation.
#                  Defaults to gpt_4o_mini.
#         """
#         self.llm = llm
#         logger.info("ReportWriterNode initialized with %s model for report writing", llm)

#     def write_report(
#         self, 
#         query: str,
#         findings: str
#     ) -> Tasks:
#         """
#         Generate a comprehensive report based on search findings.
        
#         This method synthesizes the accumulated findings into a coherent report
#         that directly addresses the original query. It formats the information in a
#         structured manner for clear presentation.
        
#         Args:
#             query: The original search query that prompted the findings.
#             findings: Accumulated findings from the search process to be synthesized.
            
#         Returns:
#             Tasks: A structured object containing the generated report.
#         """
#         logger.debug("Writing report for query: '%s' with %d characters of findings", 
#                     query[:50] + "..." if len(query) > 50 else query, 
#                     len(findings))
        
#         user_prompt = fill_prompt(
#             ReportWriterPrompts.user_prompt_format,
#             query=query,
#             findings=findings
#         )


#         messages = [
#             SystemMessage(content=ReportWriterPrompts.system_prompt),
#             HumanMessage(content=user_prompt)
#         ]

#         logger.debug("Sending report generation request to language model")
#         tasks = self.llm.invoke(messages).content
#         logger.info("Report generation completed successfully")
        
#         return tasks

#     def __call__(
#         self,
#         state: SearchState
#     ) -> SearchState:
#         """
#         Process the current search state and generate a final report.
        
#         This method serves as the main entry point for the ReportWriterNode's report
#         generation functionality. It takes the current search state, generates a
#         comprehensive report based on accumulated findings, and updates the state
#         with the generated report.
        
#         Args:
#             state: The current SearchState containing query and accumulated findings.
            
#         Returns:
#             SearchState: The updated state with the generated report added to messages.
#         """
#         logger.info("Generating final report for search query: '%s'", 
#                    state.query[:50] + "..." if len(state.query) > 50 else state.query)
        
#         consolidated_findings = "-----\n\n".join(state.findings)
#         logger.debug("Consolidated %d findings (%d total characters) for report generation", 
#                     len(state.findings), len(consolidated_findings))
        
#         answer = self.write_report(
#             state.query,
#             consolidated_findings
#         )
#         logger.info("Report successfully generated (%d characters)", len(answer))

#         state.messages += [AIMessage(content=answer)]
#         logger.debug("Search state updated with final report")
#         return state