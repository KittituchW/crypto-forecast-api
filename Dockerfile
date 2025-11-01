# ---------- Base Image ----------
FROM python:3.11.4-slim

# ---------- Set working directory ----------
WORKDIR /app

# ---------- Install system dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy dependency files ----------
COPY pyproject.toml requirements.txt* ./

# ---------- Install dependencies ----------
RUN pip install --upgrade pip
RUN if [ -f "pyproject.toml" ]; then \
        pip install poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-root; \
    elif [ -f "requirements.txt" ]; then \
        pip install -r requirements.txt; \
    fi

# ---------- Copy project files ----------
COPY app ./app
COPY models ./models

# ---------- Expose port ----------
EXPOSE 8000

CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
