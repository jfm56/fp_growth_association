# FP-Growth Association Mining

This project implements the FP-Growth algorithm for association rule mining on CSV transaction data. It includes:
- FP-Growth algorithm with FP-Tree visualization
- Docker support
- GitHub Actions workflow for CI (pytest, pylint, coverage)
- Pytest unit tests
- Pylint linting

## Usage

1. Place your CSV file in the `data/` directory.
2. Run the main script or use Docker to process the file and visualize the FP-Tree.

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
