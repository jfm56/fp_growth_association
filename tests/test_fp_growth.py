import pytest
from src.fp_growth import FPGrowth

def test_fp_growth_basic():
    transactions = [
        ['a', 'b', 'c'],
        ['a', 'b'],
        ['b', 'c'],
        ['a', 'c'],
        ['b', 'c']
    ]
    model = FPGrowth(min_support=2)
    model.fit(transactions)
    itemsets = model.get_frequent_itemsets()
    # Should find at least these frequent itemsets
    assert (['b'], 4) in itemsets
    assert (['c'], 4) in itemsets
    assert (['a'], 3) in itemsets
