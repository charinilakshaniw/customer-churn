import os
import time

import pandas as pd
import plaid
from dotenv import load_dotenv
from plaid.api import plaid_api
from plaid.api_client import ApiClient
from plaid.configuration import Configuration
from plaid.model.item_public_token_exchange_request import (
    ItemPublicTokenExchangeRequest,
)
from plaid.model.products import Products
from plaid.model.sandbox_public_token_create_request import (
    SandboxPublicTokenCreateRequest,
)
from plaid.model.transactions_refresh_request import TransactionsRefreshRequest
from plaid.model.transactions_sync_request import TransactionsSyncRequest

# -----------------------------------------
# Load environment variables
# -----------------------------------------

load_dotenv()

PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")


# -----------------------------------------
# Configure Plaid Sandbox
# -----------------------------------------

configuration = Configuration(
    host=plaid.Environment.Sandbox,
    api_key={
        "clientId": PLAID_CLIENT_ID,
        "secret": PLAID_SECRET,
    },
)

api_client = ApiClient(configuration)

plaid_client = plaid_api.PlaidApi(api_client)


# -----------------------------------------
# 1. Create Sandbox Item
# -----------------------------------------

request = SandboxPublicTokenCreateRequest(
    institution_id="ins_109508",
    initial_products=[Products("transactions")],
    options={
        "override_username": "user_transactions_dynamic",
        "override_password": "password123",
    },
)

response = plaid_client.sandbox_public_token_create(request)

public_token = response["public_token"]

print("Public token created")


# -----------------------------------------
# 2. Exchange public token
#    for access token
# -----------------------------------------

exchange_request = ItemPublicTokenExchangeRequest(public_token=public_token)

exchange_response = plaid_client.item_public_token_exchange(exchange_request)

access_token = exchange_response["access_token"]

print("Access token created")


# -----------------------------------------
# 3. Initialize transaction sync
# -----------------------------------------

transactions_request = TransactionsSyncRequest(access_token=access_token)

transactions_response = plaid_client.transactions_sync(transactions_request)

data = transactions_response.to_dict()

print("\nInitial transaction status:")
print(data["transactions_update_status"])

print("Initial transactions:")
print(len(data["added"]))


# -----------------------------------------
# 4. Refresh Sandbox transactions
# -----------------------------------------

refresh_request = TransactionsRefreshRequest(access_token=access_token)

plaid_client.transactions_refresh(refresh_request)

print("\nTransaction refresh requested")


# -----------------------------------------
# 5. Wait for Plaid
# -----------------------------------------

time.sleep(5)


# -----------------------------------------
# 6. Fetch transactions again
# -----------------------------------------

transactions_response = plaid_client.transactions_sync(transactions_request)

data = transactions_response.to_dict()

print("\nFinal transaction status:")
print(data["transactions_update_status"])

print("Number of transactions:")
print(len(data["added"]))


# -----------------------------------------
# 7. Print transactions
# -----------------------------------------

df = pd.DataFrame(data["added"])

print("\nDataFrame shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 transactions:")
print(
    df[["date", "name", "merchant_name", "amount", "category", "payment_channel"]].head(
        10
    )
)
