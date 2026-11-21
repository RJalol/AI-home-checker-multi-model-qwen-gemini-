# One image for the API, the RQ worker, the reconciler and the retention job (the command differs).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production

WORKDIR /app

# Exact, reproducible versions (requirements.lock is generated from requirements.txt with pip-compile).
COPY requirements.lock ./
RUN pip install --no-deps -r requirements.lock

COPY . .

# Never run as root; uploads/reports/training_data live on a volume owned by this user.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/uploads /app/reports /app/training_data \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
