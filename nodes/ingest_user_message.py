from typing import Dict, Any
from langchain_core.messages import HumanMessage
from state import GraphState  # adjust import to where your State class lives


def ingest_user_message(state: GraphState) -> Dict[str, Any]:
    """
    Sync the latest user message into `user_query` and extend `conversation_history`.

    This node is meant to be the first step in the graph after receiving new messages.
    It does NOT call any tools or LLMs; it's just state wiring.
    """
    user_text: str = state.user_query  # fallback, in case we don't find a HumanMessage

    # Find the last HumanMessage in the messages list (starting from the end)
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            # msg.content can be str or more complex; here we assume simple text
            user_text = msg.content
            break

    # Extend conversation history with the new user text
    conversation_history = list(state.conversation_history)
    if user_text:
        conversation_history.append(user_text)

    update: Dict[str, Any] = {
        "user_query": user_text,
        "conversation_history": conversation_history,
    }

    # Clear stale execution state at the start of every fresh query so that
    # debug fields from a previous turn don't bleed into the next response.
    # Skip this when we're mid-clarification (pending_clarification_for is set)
    # because in that case store_clarification will handle cleanup.
    if not state.pending_clarification_for:
        update.update({
            "sql_query": None,
            "sql_explanation": None,
            "execution_error": None,
            "query_result": None,
            "query_metadata": None,
            "result_status": None,
            "fix_attempts": 0,
            "assumptions": None,
            "final_answer": None,
            "clarification_question": None,
        })

    return update
