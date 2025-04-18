import logging
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from .fp_tree import FPTree

import os
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Log file in project root
log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fp_growth.log')
file_handler = RotatingFileHandler(log_path, maxBytes=1024*1024, backupCount=3)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler for errors and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# Avoid duplicate handlers
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
else:
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

class FPGrowth:
    def __init__(self, min_support: int):
        self.min_support = min_support
        self.freq_itemsets: List[Tuple[List[Any], int]] = []

    def fit(self, transactions: List[List[Any]]):
        logger.info('Starting FPGrowth.fit with %d transactions', len(transactions))
        if not transactions:
            logger.error('Transactions cannot be empty')
            raise ValueError("Transactions cannot be empty")
        
        # Count item frequency
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        logger.debug('Item counts: %s', dict(item_counts))
        # Remove infrequent items
        items = {item for item, count in item_counts.items() if count >= self.min_support}
        logger.debug('Items meeting min_support (%d): %s', self.min_support, items)
        if not items:
            logger.error('No items meet the minimum support')
            raise ValueError("No items meet the minimum support")
        
        # Filter and sort transactions
        filtered = []
        for transaction in transactions:
            filtered_tr = [item for item in transaction if item in items]
            filtered_tr.sort(key=lambda x: (-item_counts[x], x))
            if filtered_tr:
                filtered.append(filtered_tr)
        logger.debug('Filtered transactions: %s', filtered)
        
        # Build FP-Tree
        tree = FPTree()
        for idx, tr in enumerate(filtered):
            logger.debug('Inserting transaction %d: %s', idx, tr)
            tree.insert_transaction(tr)
            logger.debug('Inserted transaction %d', idx)
        logger.info('FP-Tree built. Starting mining.')
        self._mine(tree, [], item_counts, self.min_support)
        self.tree = tree
        logger.info('FPGrowth.fit complete. Found %d frequent itemsets.', len(self.freq_itemsets))

    @staticmethod
    def calculate_support(node):
        support = 0
        current = node
        while current:
            support += current.count
            current = current.link
        logger.debug('Calculated support: %d', support)
        return support

    def _mine(self, tree: FPTree, prefix: List[Any], item_counts: Dict[Any, int], min_support: int):
        logger.info('Mining FP-Tree with prefix: %s', prefix)
        if not tree.headers:
            logger.error('Tree headers cannot be empty')
            raise ValueError("Tree headers cannot be empty")
        
        # Simple mining: collect frequent items
        for item, node in tree.headers.items():
            support = self.calculate_support(node)
            logger.debug('Item: %s, Support: %d', item, support)
            if support >= min_support:
                self.freq_itemsets.append((prefix + [item], support))
                logger.info('Frequent itemset found: %s, support: %d', prefix + [item], support)

    def get_frequent_itemsets(self):
        return self.freq_itemsets
