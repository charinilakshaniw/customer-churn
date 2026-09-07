from services.customer_service import get_customer_profile
from rag.rag_service import retrieve_business_knowledge
from services.gemini_service import generate_customer_recommendation


customer_id = "C0001"


profile = get_customer_profile(customer_id)


query = f"""
What retention strategy should be used for a customer
with the following churn profile?

{profile}
"""


business_knowledge = retrieve_business_knowledge(
    query,
    k=3
)


recommendation = generate_customer_recommendation(
    customer_profile=profile,
    business_knowledge=business_knowledge
)


print("\n")
print("=" * 60)
print("CUSTOMERIQ AI RECOMMENDATION")
print("=" * 60)
print(recommendation)
print("=" * 60)