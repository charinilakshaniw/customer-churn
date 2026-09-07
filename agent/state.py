from typing import TypedDict


class CustomerIQState(TypedDict, total=False):
    user_query: str
    interaction_id: str
    pending_tool_calls: list
    function_results: list
    tools_used: list
    final_response: str
    error: str