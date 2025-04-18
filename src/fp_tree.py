from collections import defaultdict
from typing import Any, Dict, List, Optional
import logging
logger = logging.getLogger('src.fp_growth')

class FPTreeNode:
    def __init__(self, item: Any, count: int, parent: Optional['FPTreeNode'] = None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children: Dict[Any, 'FPTreeNode'] = {}
        self.link: Optional['FPTreeNode'] = None

    def add_child(self, item: Any) -> tuple['FPTreeNode', bool]:
        logger.debug('add_child called on node %s with item %s', self.item, item)
        if item not in self.children:
            self.children[item] = FPTreeNode(item, 1, self)
            logger.debug('Created new child node for item %s', item)
            return self.children[item], True
        else:
            self.children[item].count += 1
            logger.debug('Incremented count for child node %s to %d', item, self.children[item].count)
            return self.children[item], False

class FPTree:
    def __init__(self):
        self.root = FPTreeNode(None, 1)
        self.headers: Dict[Any, FPTreeNode] = {}

    def insert_transaction(self, transaction: List[Any]):
        logger.debug('FPTree.insert_transaction called with: %s', transaction)
        node = self.root
        for idx, item in enumerate(transaction):
            logger.debug('Adding item %d/%d: %s', idx+1, len(transaction), item)
            node, is_new = node.add_child(item)
            if item not in self.headers:
                self.headers[item] = node
                logger.debug('Header for %s set', item)
            elif is_new:
                current = self.headers[item]
                visited = set()
                while current.link:
                    if id(current) in visited:
                        logger.error('Cycle detected in header links for item %s', item)
                        break
                    visited.add(id(current))
                    current = current.link
                if current is not node:
                    current.link = node
                    logger.debug('Linked header for %s', item)
                else:
                    logger.error('Attempted to link node to itself for item %s', item)
        logger.debug('FPTree.insert_transaction complete for: %s', transaction)

    def visualize(self, filename: str = 'fp_tree.gv'):
        from graphviz import Digraph
        dot = Digraph(comment='FP-Tree')
        def add_nodes(node, parent_id=None):
            node_id = str(id(node))
            label = f'{node.item}:{node.count}' if node.item else 'Root'
            dot.node(node_id, label)
            if parent_id:
                dot.edge(parent_id, node_id)
            for child in node.children.values():
                add_nodes(child, node_id)
        add_nodes(self.root)
        dot.render(filename, view=True)
