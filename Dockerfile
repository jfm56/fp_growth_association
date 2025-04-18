FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
CMD ["python", "src/main.py", "data/your_file.csv"]
