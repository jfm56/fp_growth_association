import sys
from src.csv_loader import CSVLoader
from src.fp_growth import FPGrowth
from src.association_rules import AssociationRuleMiner

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "print_2itemsets":
        from itertools import combinations
        from collections import Counter
        csv_path = sys.argv[2] if len(sys.argv) > 2 else "PROJECT2_DATASET.csv"
        diagnosis_col = sys.argv[3] if len(sys.argv) > 3 else "diagnosis"
        from src.csv_loader import CSVLoader
        loader = CSVLoader(csv_path)
        transactions = loader.load_transactions(diagnosis_col=diagnosis_col)
        pair_counter = Counter()
        for tr in transactions:
            for pair in combinations(sorted(tr), 2):
                pair_counter[pair] += 1
        print("Support for all 2-itemsets (sorted by support):")
        for (a, b), count in pair_counter.most_common():
            print(f"({a}, {b}): {count}")
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "correlation":
        import pandas as pd
        csv_path = sys.argv[2] if len(sys.argv) > 2 else "PROJECT2_DATASET.csv"
        diagnosis_col = sys.argv[3] if len(sys.argv) > 3 else "diagnosis"
        df = pd.read_csv(csv_path)
        # Encode diagnosis: M=1, B=0
        df = df.copy()
        df[diagnosis_col] = df[diagnosis_col].map({"M": 1, "B": 0})
        # Compute correlation for each feature (exclude id and diagnosis)
        feature_cols = [col for col in df.columns if col not in ("id", diagnosis_col)]
        corrs = {}
        for col in feature_cols:
            try:
                corrs[col] = abs(df[col].corr(df[diagnosis_col]))
            except Exception:
                corrs[col] = float('nan')
        top_10 = sorted(corrs.items(), key=lambda x: (x[1] if x[1] == x[1] else -1), reverse=True)[:10]
        print("Top 10 features most correlated with diagnosis (absolute Pearson correlation):")
        for i, (col, corr) in enumerate(top_10, 1):
            print(f"{i}. {col}: {corr:.4f}")
        sys.exit(0)
    if len(sys.argv) < 3:
        print("Usage: python main.py <csv_file> <diagnosis_col> [min_support] [min_lift] [min_confidence] [sort_metric] [top_n]")
        sys.exit(1)
    csv_path = sys.argv[1]
    diagnosis_col = sys.argv[2]
    min_support = float(sys.argv[3]) if len(sys.argv) > 3 else 2
    min_lift = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    min_confidence = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    sort_metric = sys.argv[6] if len(sys.argv) > 6 else 'confidence'
    top_n = int(sys.argv[7]) if len(sys.argv) > 7 else 10

    loader = CSVLoader(csv_path)
    transactions = loader.load_transactions(diagnosis_col=diagnosis_col)
    print("\nSample of parsed transactions (first 5):")
    for idx, tr in enumerate(transactions[:5], 1):
        print(f"{idx}. {tr}")
    model = FPGrowth(min_support)
    model.fit(transactions)
    # Get all frequent itemsets, sort by support descending
    frequent_itemsets = sorted(model.get_frequent_itemsets(), key=lambda x: x[1], reverse=True)

    # Print all frequent itemsets of size 2 or more (no filtering)
    print("\nALL Frequent Itemsets (size >= 2, no filtering):")
    all_large_itemsets = [(itemset, support) for itemset, support in frequent_itemsets if len(itemset) >= 2]
    for idx, (itemset, support) in enumerate(all_large_itemsets, 1):
        print(f"{idx}. {itemset} - support: {support}")

    # Print all frequent single-item sets (size 1)
    print("\nFrequent Single-Item Sets (size = 1):")
    single_itemsets = [(itemset, support) for itemset, support in frequent_itemsets if len(itemset) == 1]
    for idx, (itemset, support) in enumerate(single_itemsets, 1):
        print(f"{idx}. {itemset} - support: {support}")

    # Focus output on association rules only (no raw frequent itemsets)
    # Exclude itemsets containing any diagnosis=... item and keep only itemsets of size >=2
    filtered_itemsets = [(itemset, support) for itemset, support in frequent_itemsets if len(itemset) >= 2 and not any(str(x).startswith('diagnosis=') for x in itemset)]

    # Print multi-item (2- and 3-item) frequent itemsets (excluding diagnosis)
    multi_itemsets = [(itemset, support) for itemset, support in filtered_itemsets if 2 <= len(itemset) <= 3]
    print(f"\nFrequent 2- and 3-itemsets (excluding diagnosis):")
    for idx, (itemset, support) in enumerate(multi_itemsets, 1):
        print(f"{idx}. {itemset} - support: {support}")

    # Generate association rules
    miner = AssociationRuleMiner(model.get_frequent_itemsets(), min_confidence=min_confidence, min_lift=min_lift)
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

    # Print multi-item (2- and 3-item antecedent) rules for each diagnosis
    # Only allow rules where the antecedent does not contain diagnosis and consequent is diagnosis=M
    multi_item_rules_M = [rule for rule in sorted_rules_M if 2 <= len(rule['antecedent']) <= 3 and not any(str(x).startswith('diagnosis=') for x in rule['antecedent'])]
    print(f"\nMulti-item (2- and 3-feature) rules with consequent diagnosis=M (excluding diagnosis in antecedent):")
    for i, rule in enumerate(multi_item_rules_M, 1):
        ant = ', '.join(rule['antecedent'])
        cons = ', '.join(rule['consequent'])
        print(f"{i}. IF [{ant}] THEN [{cons}] | support: {rule['support']:.3f}, conf: {rule['confidence']:.2f}, lift: {rule['lift']:.2f}")

    # Visualize FP-Tree
    model.tree.visualize("fp_tree.gv")
