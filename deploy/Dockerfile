FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2

COPY pyproject.toml README.md ./
COPY danyai ./danyai
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY runs ./runs
COPY data/tokenizer.json ./data/tokenizer.json
COPY data/meta.json ./data/meta.json

EXPOSE 8080
CMD ["python", "-m", "danyai.server", "--ckpt", "runs/sft/best.npz", "--tok", "data/tokenizer.json", "--host", "0.0.0.0", "--port", "8080"]
