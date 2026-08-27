from plaid_client import plaid_client

from plaid.model.institutions_get_request import InstitutionsGetRequest


request = InstitutionsGetRequest(
    country_codes=["US"],
    count=5,
    offset=0
)

response = plaid_client.institutions_get(request)

print(response.to_dict())