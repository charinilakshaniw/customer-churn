import numpy as np
import pandas as pd


MERCHANTS = {
    "Food and Drink": [
        "McDonald's",
        "Starbucks",
        "Uber Eats",
        "Tesco",
        "Dunkin'",
        "Pizza Hut",
    ],
    "Shopping": ["Amazon", "Walmart", "Target", "Apple", "Nike"],
    "Transport": ["Uber", "Shell", "American Airlines", "PayByPhone"],
    "Entertainment": ["Netflix", "Spotify", "Xbox", "Topgolf"],
    "Bills": ["Vodafone", "British Gas", "Microsoft"],
}


def generate_transactions(
    customers, start_date="2025-01-01", end_date="2025-12-31", seed=42
):

    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    transactions = []

    transaction_id = 1

    for _, customer in customers.iterrows():
        customer_id = customer["customer_id"]

        # -----------------------------------------
        # Assign customer behaviour
        # -----------------------------------------

        customer_type = np.random.choice(
            ["loyal", "regular", "low_engagement", "churning"],
            p=[0.40, 0.30, 0.15, 0.15],
        )

        # -----------------------------------------
        # Determine how long the customer remains
        # active
        # -----------------------------------------

        if customer_type == "loyal":
            active_until = pd.Timestamp(end_date)

            transactions_per_month = 15

        elif customer_type == "regular":
            active_until = pd.Timestamp(end_date)

            transactions_per_month = 10

        elif customer_type == "low_engagement":
            active_until = pd.Timestamp(end_date)

            transactions_per_month = 5

        else:
            # Churning customers stop before
            # the churn observation period

            active_until = pd.Timestamp(
                np.random.choice(pd.date_range("2025-06-01", "2025-09-15"))
            )

            transactions_per_month = 10

        # -----------------------------------------
        # Generate transactions
        # -----------------------------------------

        active_dates = dates[dates <= active_until]

        if len(active_dates) == 0:
            continue

        # Approximate number of transactions
        months_active = max(1, (active_dates.max() - active_dates.min()).days // 30)

        n_transactions = max(5, int(months_active * transactions_per_month))

        # Make sure we don't request more dates
        # than available
        n_transactions = min(n_transactions, len(active_dates))

        customer_dates = np.random.choice(
            active_dates, size=n_transactions, replace=False
        )

        for date in customer_dates:
            category = np.random.choice(list(MERCHANTS.keys()))

            merchant = np.random.choice(MERCHANTS[category])

            amount = round(np.random.lognormal(mean=3.2, sigma=0.8), 2)

            transactions.append(
                {
                    "transaction_id": f"T{transaction_id:07d}",
                    "customer_id": customer_id,
                    "date": date,
                    "amount": amount,
                    "merchant": merchant,
                    "category": category,
                    "payment_channel": np.random.choice(["online", "in store"]),
                    "customer_type": customer_type,
                }
            )

            transaction_id += 1

    return pd.DataFrame(transactions)


if __name__ == "__main__":
    customers = pd.DataFrame({"customer_id": [f"C{i:04d}" for i in range(1, 501)]})

    transactions = generate_transactions(customers)

    transactions.to_csv("data/raw/transactions.csv", index=False)

    print(f"Transactions: {len(transactions):,}")

    print(f"Customers: {transactions['customer_id'].nunique()}")

    print("\nCustomer types:")

    print(
        transactions[["customer_id", "customer_type"]]
        .drop_duplicates()["customer_type"]
        .value_counts()
    )
