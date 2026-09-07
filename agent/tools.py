from services.customer_service import get_customer_profile
from rag.rag_service import retrieve_business_knowledge


def get_customer_data(customer_id: str) -> dict:
    """
    Get the complete CustomerIQ profile for a customer,
    including churn prediction and SHAP explanation.
    """

    return get_customer_profile(customer_id)


def search_business_knowledge(query: str) -> list:
    """
    Search the CustomerIQ business retention playbook
    for relevant customer retention guidance.
    """

    return retrieve_business_knowledge(
        query,
        k=3
    )


CUSTOMERIQ_TOOLS = [
    {
        "type": "function",
        "name": "get_customer_data",
        "description": (
            "Retrieve a customer's profile, churn probability, "
            "risk segment, behavioural features and SHAP explanation. "
            "Use this whenever the user asks about a specific customer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": (
                        "Customer ID such as C0001"
                    )
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "type": "function",
        "name": "search_business_knowledge",
        "description": (
            "Search the CustomerIQ retention playbook for "
            "business guidance about churn, customer engagement "
            "and retention strategies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Business question to search for"
                    )
                }
            },
            "required": ["query"]
        }
    }
]


TOOL_FUNCTIONS = {
    "get_customer_data": get_customer_data,
    "search_business_knowledge": search_business_knowledge
}