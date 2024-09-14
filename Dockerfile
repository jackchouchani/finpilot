FROM python:3.12.5 AS builder
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

FROM python:3.12.5-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .
ENV PATH="/app/.venv/bin:$PATH"
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:8080"]