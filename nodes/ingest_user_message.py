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

    # In LangGraph, nodes typically return a *partial state* dict to be merged
    return {
        "user_query": user_text,
        "conversation_history": conversation_history,
    }
