# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry matching version range
RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"

# Configure Poetry to create virtual environment inside project folder
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy project specification files
COPY pyproject.toml poetry.lock ./

# Install main production dependencies (no dev dependencies, no root package install)
RUN poetry install --only main --no-root --no-directory && rm -rf $POETRY_CACHE_DIR

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Enable virtual environment by prepending it to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Set python path to point to the source code folder
ENV PYTHONPATH="/app/ecommerce_recommender/src"

# Copy application code and configs
COPY ecommerce_recommender /app/ecommerce_recommender

# Set working directory to ecommerce_recommender so relative paths resolve correctly
WORKDIR /app/ecommerce_recommender

# Default command to run training
CMD ["python", "-m", "recommender.pipelines.train_pipeline"]
