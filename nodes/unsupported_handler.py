# nodes/unsupported_handler.py

from typing import Dict, Any
from state import GraphState


def unsupported_handler(state: GraphState) -> Dict[str, Any]:
    """
    Handle requests that are not supported by the agent,
    such as write operations or semantic instructions outside scope.
    """

    user_query = (state.user_query or "").strip()
    qtype = state.question_type or "unsupported"

    msg = (
        "Sorry — I can't perform this type of request.\n\n"
        f"**Detected classification:** `{qtype}`\n"
        "This agent only supports:\n"
        "- Answering questions by reading data from the database\n"
        "- Explaining the database schema\n\n"
        "Write operations (INSERT, UPDATE, DELETE, DROP), destructive actions, or tasks\n"
        "outside the scope of database Q&A are not allowed.\n\n"
        f"Your request was: *{user_query}*"
    )

    return {"final_answer": msg}
