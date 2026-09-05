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

    transactions = []
    transaction_id = 1

    for _, customer in customers.iterrows():

        customer_id = customer["customer_id"]

        customer_type = np.random.choice(
            [
                "loyal",
                "regular",
                "low_engagement",
                "churning",
                "returning",
            ],
            p=[0.35, 0.25, 0.15, 0.15, 0.10],
        )

        # --------------------------------------------------------
        # CUSTOMER BEHAVIOUR
        # --------------------------------------------------------

        if customer_type == "loyal":

            active_periods = [
                ("2025-01-01", "2025-12-31")
            ]
            transactions_per_month = 15

        elif customer_type == "regular":

            active_periods = [
                ("2025-01-01", "2025-12-31")
            ]
            transactions_per_month = 10

        elif customer_type == "low_engagement":

            active_periods = [
                ("2025-01-01", "2025-12-31")
            ]
            transactions_per_month = 5

        elif customer_type == "churning":

            # Customer stops transacting before the observation period ends
            churn_date = pd.Timestamp(
                np.random.choice(
                    pd.date_range("2025-07-01", "2025-09-15")
                )
            )

            active_periods = [
                ("2025-01-01", str(churn_date.date()))
            ]

            transactions_per_month = 8

        else:

            # ----------------------------------------------------
            # RETURNING CUSTOMER
            # ----------------------------------------------------
            # Customer becomes inactive temporarily but returns.
            # This makes the churn problem more realistic.

            first_period_end = pd.Timestamp(
                np.random.choice(
                    pd.date_range("2025-06-01", "2025-07-31")
                )
            )

            return_date = first_period_end + pd.Timedelta(
                days=np.random.randint(45, 90)
            )

            active_periods = [
                ("2025-01-01", str(first_period_end.date())),
                (str(return_date.date()), "2025-12-31"),
            ]

            transactions_per_month = 8

        # --------------------------------------------------------
        # GENERATE TRANSACTIONS FOR EACH ACTIVE PERIOD
        # --------------------------------------------------------

        for period_start, period_end in active_periods:

            dates = pd.date_range(
                start=period_start,
                end=period_end,
                freq="D"
            )

            if len(dates) == 0:
                continue

            months_active = max(
                1,
                (dates.max() - dates.min()).days // 30
            )

            n_transactions = max(
                5,
                int(months_active * transactions_per_month)
            )

            n_transactions = min(
                n_transactions,
                len(dates)
            )

            customer_dates = np.random.choice(
                dates,
                size=n_transactions,
                replace=False
            )

            for date in customer_dates:

                category = np.random.choice(
                    list(MERCHANTS.keys())
                )

                merchant = np.random.choice(
                    MERCHANTS[category]
                )

                amount = round(
                    np.random.lognormal(
                        mean=3.2,
                        sigma=0.8
                    ),
                    2
                )

                transactions.append(
                    {
                        "transaction_id": f"T{transaction_id:07d}",
                        "customer_id": customer_id,
                        "date": date,
                        "amount": amount,
                        "merchant": merchant,
                        "category": category,
                        "payment_channel": np.random.choice(
                            ["online", "in store"]
                        ),
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
