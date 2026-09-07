import os

from dotenv import load_dotenv
from google import genai

from agent.tools import CUSTOMERIQ_TOOLS


load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set."
    )


client = genai.Client(
    api_key=API_KEY
)


MODEL_NAME = "gemini-3.6-flash"


SYSTEM_INSTRUCTIONS = """
You are CustomerIQ, an AI customer retention
decision-support agent.

You help business users understand customer
churn risk and decide appropriate retention actions.

You have access to two tools:

1. get_customer_data
   Retrieves customer behavioural data,
   churn prediction and SHAP explanations.

2. search_business_knowledge
   Retrieves relevant business retention guidance.

Rules:

- When a specific customer is mentioned,
  retrieve their customer data before making
  a recommendation.

- Use the business knowledge tool when a
  retention recommendation is required.

- Never invent customer information.

- Use the ML prediction and SHAP explanation
  to explain risk.

- Use the business knowledge retrieved from
  the retention playbook to recommend actions.

- Consider both churn risk and customer value.

- Do not recommend aggressive discounts to
  low-risk customers unless the business
  knowledge explicitly supports it.

- Give concise, business-focused answers.

Structure final answers as:

Risk Assessment

Key Drivers

Recommended Action

Reasoning
"""


def start_agent(
    user_query: str
):

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": user_query
                    }
                ]
            }
        ],
        tools=CUSTOMERIQ_TOOLS,
        system_instruction=SYSTEM_INSTRUCTIONS,
        generation_config={
            "thinking_level": "low"
        }
    )

    return interaction


def continue_agent(
    interaction_id: str,
    function_results: list
):

    interaction = client.interactions.create(
        model=MODEL_NAME,
        previous_interaction_id=interaction_id,
        input=function_results,
        tools=CUSTOMERIQ_TOOLS,
        system_instruction=SYSTEM_INSTRUCTIONS,
        generation_config={
            "thinking_level": "low"
        }
    )

    return interaction