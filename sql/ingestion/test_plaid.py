from plaid.model.institutions_get_request import InstitutionsGetRequest
from plaid_client import plaid_client

request = InstitutionsGetRequest(country_codes=["US"], count=5, offset=0)

response = plaid_client.institutions_get(request)

print(response.to_dict())
