import pandas as pd
from typing import List

class CSVLoader:
    def __init__(self, path: str):
        self.path = path

    def load_transactions(self, diagnosis_col: str = None) -> List[List[str]]:
        df = pd.read_csv(self.path, header=0)
        # Only include the top 10 correlated features (from previous step) plus diagnosis
        top_features = [
            'concave points_worst',
            'perimeter_worst',
            'concave points_mean',
            'radius_worst',
            'perimeter_mean',
            'area_worst',
            'radius_mean',
            'area_mean',
            'concavity_mean',
            'concavity_worst'
        ]
        feature_cols = [col for col in df.columns if col in top_features]
        # Compute median split bins for each feature
        bin_labels = ['L', 'H']
        bins = {}
        for col in feature_cols:
            try:
                bins[col] = pd.qcut(df[col], q=2, labels=bin_labels, duplicates='drop')
            except ValueError:
                bins[col] = pd.cut(df[col], bins=2, labels=bin_labels, duplicates='drop')
        transactions = []
        for idx, row in df.iterrows():
            items = []
            # Only include diagnosis and top features
            for col in [diagnosis_col] + top_features:
                if col not in df.columns:
                    continue
                value = row[col]
                if pd.isna(value):
                    continue
                if col == diagnosis_col:
                    items.append(f"{col}={value}")
                elif col in bins:
                    bin_label = bins[col].iloc[idx]
                    items.append(f"{col}={bin_label}")
            transactions.append(items)
        return transactions

