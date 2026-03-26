from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage

from state import GraphState


class SQLPlan(BaseModel):
    sql_query: str
    sql_explanation: str
    assumptions: Optional[List[str]] = None


def make_nl_to_sql_planner_node(llm: BaseChatModel):
    """
    Factory that creates an nl_to_sql_planner node bound to a specific LLM.

    Uses structured output (SQLPlan) to generate a SQL query + explanation.
    """

    structured_llm = llm.with_structured_output(SQLPlan)

    def nl_to_sql_planner(state: GraphState) -> Dict[str, Any]:
        """
        Convert the natural language user_query into a SQL query.

        Reads:  user_query, db_schema_summary, conversation_history
        Writes: sql_query, sql_explanation, assumptions
        """
        user_query = (state.user_query or "").strip()
        schema_summary = (
            state.db_schema_summary or "No schema information available."
        )
        known_terms = state.known_terms or {}
        last_sql_query = state.last_sql_query or ""

        if not user_query:
            return {
                "sql_query": "",
                "sql_explanation": "No user query provided.",
                "assumptions": [],
            }

        history_snippets = (
            state.conversation_history[-5:]
            if state.conversation_history else []
        )
        history_text = (
            "\n".join(history_snippets)
            if history_snippets else "No prior conversation."
        )

        system_prompt = (
            "You are an assistant that translates natural language questions "
            "into SQL queries for a SQLite database.\n\n"
            "You will be given:\n"
            "- A description of the database schema.\n"
            "- The user's question.\n"
            "- A brief conversation history.\n"
            "- Optionally: the last successfully executed SQL query.\n\n"
            "Follow-up query rule (IMPORTANT):\n"
            "- If the user's question is a variation or modification of the "
            "previous query (e.g. 'do the same for 2012', 'change year to "
            "2014', 'sort by revenue descending', 'show only top 5'), "
            "start from the previous SQL and apply only the minimal change "
            "needed — do NOT rebuild the query from scratch.\n"
            "- If no previous SQL is provided, or the question is unrelated, "
            "write a new query from scratch.\n\n"
            "Your task is to:\n"
            "1. Understand what the user wants to retrieve.\n"
            "2. Produce a SINGLE SQL query that answers the question.\n"
            "3. ONLY use SELECT (no INSERT, UPDATE, DELETE, DROP, ALTER).\n"
            "4. Prefer explicit column lists over SELECT *.\n"
            "5. Use SQLite-compatible SQL syntax.\n\n"
            "Date filtering rules (IMPORTANT):\n"
            "- If the question does NOT mention any specific date, year, "
            "month, quarter, or time period, do NOT add a WHERE clause "
            "filtering by date — query the full table instead.\n"
            "- Only restrict by date when the user explicitly provides or "
            "implies a time range "
            "(e.g. 'in 1997', 'last month', 'Q1 2023').\n"
            "- NEVER use date('now'), datetime('now'), or any current-date "
            "function. The database is a static historical snapshot — 'now' "
            "points past the last record and returns zero rows.\n"
            "- ALWAYS replace every occurrence of date('now', ...) with:\n"
            "    date((SELECT MAX(OrderDate) FROM Orders), ...)\n"
            "  Examples:\n"
            "    date('now', '-6 months')  →  date((SELECT MAX(OrderDate) FROM Orders), '-6 months')\n"
            "    date('now', '-12 months') →  date((SELECT MAX(OrderDate) FROM Orders), '-12 months')\n"
            "    date('now', '-18 months') →  date((SELECT MAX(OrderDate) FROM Orders), '-18 months')\n"
            "  This rule applies everywhere in the query: WHERE, BETWEEN, CTEs, subqueries.\n\n"
            "Date/time function style (CRITICAL):\n"
            "- ALWAYS use PostgreSQL-style date functions: "
            "DATE_TRUNC, EXTRACT, TO_CHAR.\n"
            "- NEVER use strftime() under any circumstances — "
            "not even as a fallback. It will produce silent NULL "
            "values for quarter/year without raising an error.\n"
            "- For quarters, use: EXTRACT(QUARTER FROM col)\n"
            "- For years,    use: EXTRACT(YEAR FROM col)\n"
            "- For months,   use: EXTRACT(MONTH FROM col)\n\n"
            "Return:\n"
            "- sql_query: the SQL string\n"
            "- sql_explanation: short explanation of how it answers the "
            "question\n"
            "- assumptions: list of assumptions made (or empty list)"
        )

        known_terms_text = ""
        if known_terms:
            lines = "\n".join(f"- {k}: {v}" for k, v in known_terms.items())
            known_terms_text = f"\nDomain terminology:\n{lines}\n"

        last_sql_text = (
            f"\nPrevious SQL query (modify if this is a follow-up):\n"
            f"```sql\n{last_sql_query}\n```\n"
            if last_sql_query else ""
        )

        user_prompt = (
            "Database schema:\n"
            f"{schema_summary}\n"
            f"{known_terms_text}"
            f"{last_sql_text}\n"
            "User question:\n"
            f"{user_query}\n\n"
            "Recent conversation history:\n"
            f"{history_text}"
        )

        messages: list[AnyMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        plan: SQLPlan = structured_llm.invoke(messages)

        return {
            "sql_query": plan.sql_query,
            "sql_explanation": plan.sql_explanation,
            "assumptions": plan.assumptions or [],
        }

    return nl_to_sql_planner
