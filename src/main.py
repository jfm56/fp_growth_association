import sys
from src.csv_loader import CSVLoader
from src.fp_growth import FPGrowth
from src.association_rules import AssociationRuleMiner

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <csv_file> <diagnosis_col> [min_support] [diagnosis_value] [min_lift] [sort_metric] [top_n]")
        sys.exit(1)
    csv_path = sys.argv[1]
    diagnosis_col = sys.argv[2]
    min_support = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    min_lift = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    sort_metric = sys.argv[5] if len(sys.argv) > 5 else 'confidence'
    top_n = int(sys.argv[6]) if len(sys.argv) > 6 else 10

    loader = CSVLoader(csv_path)
    transactions = loader.load_transactions(diagnosis_col=diagnosis_col)
    model = FPGrowth(min_support)
    model.fit(transactions)
    # Get all frequent itemsets, sort by support descending
    frequent_itemsets = sorted(model.get_frequent_itemsets(), key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} Frequent Itemsets:")
    for idx, (itemset, support) in enumerate(frequent_itemsets[:top_n], 1):
        print(f"{idx}. {itemset} - support: {support}")

    # Generate association rules
    miner = AssociationRuleMiner(model.get_frequent_itemsets(), min_lift=min_lift)
    rules = miner.generate_rules()
    # Filter rules for each diagnosis value
    rules_M = miner.filter_rules_by_diagnosis(rules, 'M')
    rules_B = miner.filter_rules_by_diagnosis(rules, 'B')
    sorted_rules_M = miner.sort_rules(rules_M, metric=sort_metric)[:top_n]
    sorted_rules_B = miner.sort_rules(rules_B, metric=sort_metric)[:top_n]
    print(f"\nTop {top_n} Rules with consequent diagnosis=M and diagnosis=B (min_lift={min_lift}):\n")
    print(f"{'diagnosis=M'.center(70)} | {'diagnosis=B'.center(70)}")
    print(f"{'-'*70} | {'-'*70}")
    for i in range(top_n):
        left = right = ''
        if i < len(sorted_rules_M):
            rule = sorted_rules_M[i]
            ant = ', '.join(rule['antecedent'])
            cons = ', '.join(rule['consequent'])
            left = f"{i+1}. IF [{ant}] THEN [{cons}] | support: {rule['support']:.3f}, conf: {rule['confidence']:.2f}, lift: {rule['lift']:.2f}"
        if i < len(sorted_rules_B):
            rule = sorted_rules_B[i]
            ant = ', '.join(rule['antecedent'])
            cons = ', '.join(rule['consequent'])
            right = f"{i+1}. IF [{ant}] THEN [{cons}] | support: {rule['support']:.3f}, conf: {rule['confidence']:.2f}, lift: {rule['lift']:.2f}"
        print(f"{left.ljust(70)} | {right.ljust(70)}")

    # Visualize FP-Tree
    model.tree.visualize("fp_tree.gv")
