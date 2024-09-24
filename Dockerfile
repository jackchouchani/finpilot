# Utiliser une version plus ancienne de Python qui pourrait être plus stable avec ces bibliothèques
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Ajouter ces variables d'environnement pour la compilation
    CFLAGS="-fno-tree-vectorize" \
    CXXFLAGS="-fno-tree-vectorize"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    ca-certificates \
    fuse3 \
    sqlite3 \
    # Ajouter cmake qui est parfois nécessaire pour certaines bibliothèques
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy the LiteFS binary
COPY --from=flyio/litefs:0.5 /usr/local/bin/litefs /usr/local/bin/litefs

# Set up Python environment
RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
# Installer Cython en premier
RUN pip install --no-cache-dir cython
# Installer les dépendances en utilisant --prefer-binary
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Copy the logo file
COPY logo.jpg /app/logo.jpg

# Create necessary directories for LiteFS
RUN mkdir -p /var/lib/litefs /litefs

# Copy LiteFS configuration
COPY litefs.yml /etc/litefs.yml

# Set the command to use LiteFS
CMD ["litefs", "mount"]