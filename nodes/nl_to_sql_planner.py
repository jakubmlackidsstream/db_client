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

    It uses structured output (SQLPlan) to generate a single SQL query and an explanation.
    """

    structured_llm = llm.with_structured_output(SQLPlan)

    def nl_to_sql_planner(state: GraphState) -> Dict[str, Any]:
        """
        Convert the natural language user_query into a SQL query using the DB schema.

        Reads:
          - user_query
          - db_schema_summary
          - conversation_history (optional for follow-ups)

        Writes:
          - sql_query
          - sql_explanation
          - assumptions
        """
        user_query = (state.user_query or "").strip()
        schema_summary = state.db_schema_summary or "No schema information available."

        if not user_query:
            # Safety fallback: if for some reason we have no user_query at this point
            return {
                "sql_query": "",
                "sql_explanation": "No user query provided; cannot generate SQL.",
                "assumptions": [],
            }

        # Optional: include a bit of conversation history for follow-ups
        history_snippets = state.conversation_history[-5:] if state.conversation_history else []
        history_text = "\n".join(history_snippets) if history_snippets else "No prior conversation."

        system_prompt = (
            "You are an assistant that translates natural language questions into SQL queries "
            "for a SQLite database.\n\n"
            "You will be given:\n"
            "- A description of the database schema.\n"
            "- The user's question.\n"
            "- A brief conversation history.\n\n"
            "Your task is to:\n"
            "1. Understand what the user wants to retrieve from the database.\n"
            "2. Produce a SINGLE SQL query that answers the question.\n"
            "3. ONLY generate read-only queries using SELECT (no INSERT, UPDATE, DELETE, DROP, ALTER, etc.).\n"
            "4. Prefer explicit column lists over SELECT * when reasonable.\n"
            "5. Use SQLite-compatible SQL syntax.\n\n"
            "Use the provided response schema to return:\n"
            "- sql_query: the SQL string\n"
            "- sql_explanation: a short explanation of how this query answers the question\n"
            "- assumptions: any assumptions you had to make (or an empty list)"
        )

        user_prompt = (
            "Database schema:\n"
            f"{schema_summary}\n\n"
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