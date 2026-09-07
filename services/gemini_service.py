import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Add it to your .env file."
    )


client = genai.Client(
    api_key=API_KEY
)


def generate_response(prompt: str) -> str:

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
        generation_config={
            "thinking_level": "low"
        }
    )

    return interaction.output_text


def generate_customer_recommendation(
    customer_profile: dict,
    business_knowledge: list
) -> str:

    knowledge_text = "\n\n".join(
        [
            item["content"]
            for item in business_knowledge
        ]
    )

    prompt = f"""
You are CustomerIQ, an AI customer retention
decision-support assistant.

Your job is to analyse a customer's churn risk
and provide a concise, business-focused recommendation.

CUSTOMER PROFILE
----------------
{customer_profile}

BUSINESS KNOWLEDGE
------------------
{knowledge_text}

INSTRUCTIONS
------------
1. Explain the customer's current churn risk.
2. Identify the most important behavioural factors.
3. Recommend an appropriate retention action.
4. Consider both churn risk and customer value.
5. Do not invent information that is not provided.
6. Keep the recommendation practical and concise.

Return your response using these sections:

Risk Assessment
Key Drivers
Recommended Action
Reasoning
"""

    return generate_response(prompt)