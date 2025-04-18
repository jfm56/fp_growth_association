# FP-Growth Association Mining

This project implements the FP-Growth algorithm for association rule mining on CSV transaction data.

## Workflow and Findings

### Dataset & Preprocessing
- The dataset is a CSV with feature columns, an `id` column, and a `diagnosis` column (`M`/`B`).
- Only the 10 features most correlated with diagnosis are included in mining (see below).
- All features are binned at the median into 'L' (low) and 'H' (high) per column.
- Transactions are lists like: `[diagnosis=M, feature1=H, feature2=L, ...]`.

### Top 10 Correlated Features
The following features are most correlated with diagnosis (absolute Pearson correlation):
1. concave points_worst
2. perimeter_worst
3. concave points_mean
4. radius_worst
5. perimeter_mean
6. area_worst
7. radius_mean
8. area_mean
9. concavity_mean
10. concavity_worst

### Association Mining Results
- With all features binned and support as low as 1%, no frequent itemsets of size 2 or more were found.
- This is due to high diversity in the data, even among the most correlated features.
- Frequent single-item sets are present (e.g., `radius_mean=H`), but multi-item combinations are rare.
- The support for all possible 2-itemsets can be printed for further exploration.

### Utilities
- **Top Correlated Features:**
  ```sh
  python3 src/main.py correlation PROJECT2_DATASET.csv diagnosis
  ```
- **Print Support for All 2-Itemsets:**
  ```sh
  python3 src/main.py print_2itemsets PROJECT2_DATASET.csv diagnosis
  ```
- **Run Decision Tree (predicting M):**
  ```sh
  python3 src/main.py decision_tree PROJECT2_DATASET.csv diagnosis
  ```
  - Uses only the top 10 features most correlated with diagnosis, binned at the median (0=low, 1=high).
  - Reports accuracy, precision, recall, F1, tree depth, and a text visualization of the tree for predicting malignant (M).

### Recommendations
- For more meaningful multi-itemsets, consider alternative binning, feature engineering, or focus on categorical features.
- Use the 2-itemset support utility to identify potential pairs for further analysis.

## Development
- Run tests: `pytest`
- Lint: `pylint src/`
- Coverage: `pytest --cov=src`

## Docker
Build and run using Docker:
```sh
docker build -t fp-growth .
docker run --rm -v $(pwd)/data:/app/data fp-growth
```

## GitHub Actions
CI pipeline runs tests, lint, and coverage on push.
