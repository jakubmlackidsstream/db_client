from typing import Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage

from state import GraphState  # adjust to your actual module


def make_result_summarizer_node(llm: BaseChatModel, max_rows: int = 20, max_cols: int = 10):
    """
    Factory that creates a result_summarizer node bound to a specific LLM.

    max_rows: maximum number of rows from query_result to show in the prompt
              and in the Markdown table appended to the final answer.
    max_cols: maximum number of columns to include per row.
    """

    def _format_result_for_prompt(query_result: List[Dict[str, Any]], max_rows: int, max_cols: int) -> str:
        """
        Turn the query_result (list of dicts) into a compact text table for the LLM prompt.
        """
        if not query_result:
            return "No rows returned."

        all_columns = list(query_result[0].keys())
        columns = all_columns[:max_cols]

        lines: List[str] = []
        header = " | ".join(columns)
        lines.append(header)
        lines.append("-" * len(header))

        for row in query_result[:max_rows]:
            values = []
            for col in columns:
                val = row.get(col)
                s = str(val)
                if len(s) > 40:
                    s = s[:37] + "..."
                values.append(s)
            lines.append(" | ".join(values))

        if len(query_result) > max_rows:
            lines.append(f"... ({len(query_result) - max_rows} more rows not shown)")

        return "\n".join(lines)

    def _build_markdown_table(query_result: List[Dict[str, Any]], max_rows: int, max_cols: int) -> str:
        """
        Build a proper Markdown table to append to the final answer.
        """
        if not query_result:
            return ""

        columns = list(query_result[0].keys())[:max_cols]

        lines: List[str] = []
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")

        for row in query_result[:max_rows]:
            cells = []
            for col in columns:
                val = row.get(col)
                s = str(val) if val is not None else ""
                s = s.replace("|", "\\|")
                if len(s) > 50:
                    s = s[:47] + "..."
                cells.append(s)
            lines.append("| " + " | ".join(cells) + " |")

        if len(query_result) > max_rows:
            lines.append(f"\n*... {len(query_result) - max_rows} more rows not shown*")

        return "\n".join(lines)

    def result_summarizer(state: GraphState) -> Dict[str, Any]:
        """
        Summarize the SQL query_result into a human-friendly answer.

        Reads:
          - user_query
          - query_result
          - query_metadata
          - sql_explanation (optional)

        Writes:
          - final_answer
          - last_query
          - last_sql_query
          - last_result_summary
        """
        user_query = (state.user_query or "").strip()
        query_result = state.query_result or []
        query_metadata = state.query_metadata or {}
        sql_explanation = state.sql_explanation or ""

        row_count = query_metadata.get("row_count", len(query_result))
        columns = query_metadata.get("columns", list(query_result[0].keys()) if query_result else [])

        # If there are no rows, we still try to produce a friendly message
        if not query_result:
            final_answer = (
                f"I ran a query to answer: '{user_query}', but it returned no rows.\n"
                "This usually means there is no matching data for the filters or criteria implied by your question."
            )
            return {
                "final_answer": final_answer,
                "last_query": user_query,
                "last_sql_query": state.sql_query,
                "last_result_summary": final_answer,
            }

        result_preview = _format_result_for_prompt(query_result, max_rows, max_cols)

        system_prompt = (
            "You are an assistant that explains SQL query results in clear, concise natural language.\n\n"
            "Given:\n"
            "- The user's question\n"
            "- A preview of the tabular results (rows and columns)\n"
            "- Optional metadata (row count, column names)\n"
            "- An explanation of the SQL intent\n\n"
            "Your job is to:\n"
            "1. Answer the user's question directly based on the data.\n"
            "2. Optionally highlight key numbers, trends, or the top few items.\n"
            "3. If there are many rows, summarize the pattern instead of listing everything.\n"
            "4. Keep the answer concise and easy to understand.\n"
        )

        user_prompt = (
            f"User question:\n{user_query}\n\n"
            f"Row count: {row_count}\n"
            f"Columns: {columns}\n\n"
            "Preview of query result (truncated):\n"
            f"{result_preview}\n\n"
            f"SQL explanation (optional):\n{sql_explanation}\n\n"
            "Now, provide a natural language answer for the user."
        )

        messages: list[AnyMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        summary = response.content if hasattr(response, "content") else str(response)

        markdown_table = _build_markdown_table(query_result, max_rows, max_cols)
        final_answer = f"{summary}\n\n{markdown_table}" if markdown_table else summary

        return {
            "final_answer": final_answer,
            "last_query": user_query,
            "last_sql_query": state.sql_query,
            "last_result_summary": summary,
        }

    return result_summarizer
