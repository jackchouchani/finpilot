FROM python:3.12.5 AS builder
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

# OU pour les images basées sur debian/ubuntu
# If you're using a Debian/Ubuntu-based image
RUN apt-get update && apt-get install -y ca-certificates fuse3 sqlite3

# Copy the LiteFS binary into your container
COPY --from=flyio/litefs:0.5 /usr/local/bin/litefs /usr/local/bin/litefs
ENTRYPOINT ["litefs", "mount"]

FROM python:3.12.5-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
CMD ["gunicorn", "--timeout", "180", "wsgi:app", "--bind", "0.0.0.0:8080"]
