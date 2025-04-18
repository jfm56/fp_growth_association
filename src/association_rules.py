from typing import List, Tuple, Any, Dict
from itertools import combinations

class AssociationRuleMiner:
    def __init__(self, freq_itemsets: List[Tuple[List[Any], int]], min_confidence: float = 0.5, min_lift: float = 0.0):
        self.freq_itemsets = freq_itemsets
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.itemset_support = {tuple(sorted(itemset)): support for itemset, support in freq_itemsets}
        self.total_transactions = sum([support for _, support in freq_itemsets if len(_) == 1])

    def generate_rules(self) -> List[Dict]:
        rules = []
        for itemset, support in self.freq_itemsets:
            if len(itemset) < 2:
                continue
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = tuple(sorted(antecedent))
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))
                    if not consequent:
                        continue
                    antecedent_support = self.itemset_support.get(antecedent, 0)
                    if antecedent_support == 0:
                        continue
                    confidence = support / antecedent_support
                    consequent_support = self.itemset_support.get(consequent, 0)
                    support_ratio = support / self.total_transactions if self.total_transactions > 0 else 0
                    lift = confidence / (consequent_support / self.total_transactions) if consequent_support and self.total_transactions > 0 else 0
                    if confidence >= self.min_confidence and lift >= self.min_lift:
                        rules.append({
                            'antecedent': list(antecedent),
                            'consequent': list(consequent),
                            'support': support_ratio,
                            'confidence': confidence,
                            'lift': lift
                        })
        return rules

    @staticmethod
    def filter_rules_by_diagnosis(rules: List[Dict], diagnosis_value: str) -> List[Dict]:
        diagnosis_item = f"diagnosis={diagnosis_value}"
        return [rule for rule in rules if rule['consequent'] == [diagnosis_item]]

    @staticmethod
    def sort_rules(rules: List[Dict], metric: str = 'confidence', reverse: bool = True) -> List[Dict]:
        return sorted(rules, key=lambda x: x.get(metric, 0), reverse=reverse)

    @staticmethod
    def filter_rules_by_lift(rules: List[Dict], min_lift: float) -> List[Dict]:
        return [rule for rule in rules if rule['lift'] >= min_lift]
