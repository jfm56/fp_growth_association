import pandas as pd
from typing import List

class CSVLoader:
    def __init__(self, path: str):
        self.path = path

    def load_transactions(self, diagnosis_col: str = None) -> List[List[str]]:
        df = pd.read_csv(self.path, header=0)
        feature_cols = [col for col in df.columns if col not in ('id', diagnosis_col)]
        # Compute min, mean, max for each feature
        stats = {}
        for col in feature_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            col_mean = df[col].mean()
            stats[col] = {'min': col_min, 'mean': col_mean, 'max': col_max}
        transactions = []
        for _, row in df.iterrows():
            items = []
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    continue
                if col == 'id':
                    continue
                if col == diagnosis_col:
                    items.append(f"{col}={value}")
                else:
                    col_stats = stats[col]
                    # Find which bin value belongs to
                    diffs = {
                        'low': abs(value - col_stats['min']),
                        'mean': abs(value - col_stats['mean']),
                        'high': abs(value - col_stats['max'])
                    }
                    bin_label = min(diffs, key=diffs.get)
                    items.append(f"{col}={bin_label}")
            transactions.append(items)
        return transactions
