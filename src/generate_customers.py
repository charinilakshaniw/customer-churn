import numpy as np
import pandas as pd


def generate_customers(n_customers=500, seed=42):

    np.random.seed(seed)

    customers = pd.DataFrame(
        {"customer_id": [f"C{i:04d}" for i in range(1, n_customers + 1)]}
    )

    return customers


if __name__ == "__main__":
    df = generate_customers()

    print(df.head())
    print(f"\nNumber of customers: {len(df)}")
