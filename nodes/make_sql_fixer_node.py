from typing import Dict, Any, Optional, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from pydantic import BaseModel

from state import GraphState


class FixedSQL(BaseModel):
    sql_query: str
    sql_explanation: str
    assumptions: Optional[List[str]] = None


def make_sql_fixer_node(llm: BaseChatModel):
    """
    Factory that creates a sql_fixer node bound to a specific LLM.

    Given a failed SQL query and its execution error, asks the LLM to produce
    a corrected query, then clears the error so the executor can retry.
    """

    structured_llm = llm.with_structured_output(FixedSQL)

    def sql_fixer(state: GraphState) -> Dict[str, Any]:
        """
        Fix a broken SQL query using the execution error message.

        Reads:
          - sql_query
          - execution_error
          - user_query
          - db_schema_summary
          - fix_attempts

        Writes:
          - sql_query       (corrected)
          - sql_explanation
          - assumptions
          - fix_attempts    (incremented)
          - execution_error (cleared)
        """
        broken_sql = state.sql_query or ""
        error_message = state.execution_error or "Unknown error"
        user_query = (state.user_query or "").strip()
        schema_summary = (
            state.db_schema_summary or "No schema information available."
        )

        system_prompt = (
            "You are an expert SQL debugger for SQLite databases.\n\n"
            "You will be given:\n"
            "- A SQL query that failed to execute.\n"
            "- The error message produced by the database.\n"
            "- The database schema.\n"
            "- The original user question.\n\n"
            "Your task is to:\n"
            "1. Identify the cause of the error.\n"
            "2. Produce a corrected SQL query that avoids the error "
            "and answers the question.\n"
            "3. Only generate read-only SELECT queries "
            "(no INSERT, UPDATE, DELETE, DROP, etc.).\n"
            "4. Use SQLite-compatible syntax.\n"
            "5. NEVER use date('now') or datetime('now') — the database"
            " is a static historical snapshot, 'now' is past all records"
            " and returns zero rows. Replace every date('now', ...) "
            "with:\n"
            "   date((SELECT MAX(OrderDate) FROM Orders), ...)\n"
            "   e.g. date('now', '-6 months') ->\n"
            "        date((SELECT MAX(OrderDate) FROM Orders),"
            " '-6 months')\n"
            "6. SQLite does NOT support PostgreSQL/standard date "
            "functions. Replace them as follows:\n"
            "   DATE_TRUNC('year',  col) -> strftime('%Y', col)\n"
            "   DATE_TRUNC('month', col) -> strftime('%Y-%m', col)\n"
            "   DATE_TRUNC('quarter', col) -> not supported, use:\n"
            "     ((CAST(strftime('%m', col) AS INTEGER) - 1) / 3)"
            " + 1\n"
            "   EXTRACT(YEAR  FROM col) ->"
            " CAST(strftime('%Y', col) AS INTEGER)\n"
            "   EXTRACT(MONTH FROM col) ->"
            " CAST(strftime('%m', col) AS INTEGER)\n"
            "   TO_CHAR(col, 'YYYY-MM') -> strftime('%Y-%m', col)\n"
        )

        user_prompt = (
            f"Original user question:\n{user_query}\n\n"
            f"Database schema:\n{schema_summary}\n\n"
            f"Failed SQL query:\n{broken_sql}\n\n"
            f"Execution error:\n{error_message}\n\n"
            "Provide a corrected SQL query."
        )

        messages: list[AnyMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        fixed: FixedSQL = structured_llm.invoke(messages)  # type: ignore[assignment]

        return {
            "sql_query": fixed.sql_query,
            "sql_explanation": fixed.sql_explanation,
            "assumptions": fixed.assumptions or [],
            "fix_attempts": state.fix_attempts + 1,
            "execution_error": None,  # clear so executor runs clean
        }

    return sql_fixer
