# Slim base image -- the serving app only needs Flask + scikit-learn's
# runtime, not a full build toolchain.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, separately from application code, so Docker's
# layer cache skips this (slow) step on every rebuild unless requirements.txt
# actually changed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the pre-trained model artifacts.
COPY app.py .
COPY model/ model/

# Run as a non-root user -- a real (if minimal) container-security practice,
# not in the original lab manual's version of this Dockerfile.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "app.py"]
