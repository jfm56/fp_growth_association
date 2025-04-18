import tempfile
import pandas as pd
from src.csv_loader import CSVLoader

def test_csv_loader():
    df = pd.DataFrame([
        ['a', 'b', 'c'],
        ['a', 'b', None],
        ['b', 'c', None]
    ])
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False) as f:
        df.to_csv(f.name, header=False, index=False)
        loader = CSVLoader(f.name)
        transactions = loader.load_transactions()
        assert transactions == [['a', 'b', 'c'], ['a', 'b'], ['b', 'c']]
