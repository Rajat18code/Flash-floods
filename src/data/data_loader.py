import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def load_csv(self):
        """Load a CSV dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        df = pd.read_csv(self.data_path, skiprows=12)

        print("Dataset loaded successfully.")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        return df

    @staticmethod
    def inspect_data(df):
        """Display basic dataset information."""
        print("\n--- Dataset Info ---")
        print(df.info())

        print("\n--- First 5 Rows ---")
        print(df.head())

        print("\n--- Missing Values ---")
        print(df.isnull().sum())

        print("\n--- Statistical Summary ---")
        print(df.describe())


if __name__ == "__main__":
    data_path = "data/raw/himachal_weather.csv"

    loader = DataLoader(data_path)
    df = loader.load_csv()

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset information:")
    df.info()