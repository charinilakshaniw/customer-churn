import os

from dotenv import load_dotenv

from plaid.api import plaid_api
from plaid.configuration import Configuration
from plaid.api_client import ApiClient
from plaid.environment import Environment


load_dotenv()


PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")


configuration = Configuration(
    host=Environment.Sandbox,
    api_key={
        "clientId": PLAID_CLIENT_ID,
        "secret": PLAID_SECRET,
    },
)

api_client = ApiClient(configuration)

plaid_client = plaid_api.PlaidApi(api_client)