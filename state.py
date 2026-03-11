from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage
import operator


class GraphState(BaseModel):
    # -- Messages --
    messages: Annotated[List[AnyMessage], operator.add]

    # -- Core interaction --
    user_query: str = Field("", description="Current user message")
    conversation_history: List[str] = Field(
        default_factory=list,
        description="List of past messages or compact summaries"
    )

    # -- Classification / intent --
    question_type: Optional[str] = Field(
        default=None,
        description="db_read | db_schema_question | chitchat | unsupported",
    )
    intent_description: Optional[str] = Field(
        default=None, description="Natural language interpretation of the user question"
    )

    # -- Schema & metadata (for any database schema) --
    db_schema_raw: Optional[Any] = Field(
        default=None, description="Raw schema returned by the schema introspection tool"
    )
    db_schema: Optional[Any] = Field(
        default=None, description="Normalized or LLM-friendly schema structure"
    )
    db_schema_summary: Optional[str] = Field(
        default=None, description="Textual summary of schema"
    )
    db_schema_last_updated: Optional[str] = Field(
        default=None, description="Timestamp or version of last schema update"
    )
    db_schema_ready: Optional[bool] = Field(
        default=None, description="Whether schema is available and ready for use"
    )

    # -- SQL planning --
    sql_query: Optional[str] = Field(
        default=None, description="Generated SQL query"
    )
    sql_explanation: Optional[str] = Field(
        default=None, description="Explanation of how the SQL matches the question"
    )
    assumptions: Optional[List[str]] = Field(
        default=None, description="Assumptions made during query planning"
    )

    # -- Guardrail --
    sql_safe: Optional[bool] = Field(
        default=None, description="Whether SQL is read-only"
    )

    query_result: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Rows returned by SQL execution"
    )
    query_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata about the query result (row_count, columns, etc.)"
    )
    execution_error: Optional[str] = Field(
        default=None,
        description="SQL execution error, if any"
    )
    result_status: Optional[str] = None
    fix_attempts: int = 0
    clarification_question: Optional[str] = None

    last_query: Optional[str] = None
    last_sql_query: Optional[str] = None
    last_result_summary: Optional[str] = None
    last_filters: Optional[Dict[str, Any]] = None

    final_answer: Optional[str] = None

    # -- Knowledge store --
    known_terms: Dict[str, str] = Field(
        default_factory=dict,
        description="User-defined vocabulary: term → definition"
    )
    pending_clarification_for: Optional[str] = Field(
        default=None,
        description="Term currently awaiting user clarification"
    )
    ambiguous_terms: List[str] = Field(
        default_factory=list,
        description="Terms in the current query flagged as ambiguous"
    )
