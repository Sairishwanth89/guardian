# ─────────────────────────────────────────────────────────────────────────────
# GUARDIAN Fleet — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Builds the OpenEnv-compatible inference + demo image.
# Training (GRPO / Unsloth) is done separately on Kaggle T4 x2.
#
# Build:
#   docker build -t guardian-env -f server/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... guardian-env
#
# Then open:
#   http://localhost:8000/health   — health check
#   http://localhost:8000/web      — browser UI
#   http://localhost:8000/docs     — OpenAPI docs
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="GUARDIAN Team"
LABEL description="GUARDIAN Fleet — AI Security Oversight RL Environment (OpenEnv)"
LABEL version="0.2.0"

WORKDIR /app

# ── 1. Install system dependencies ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
  && rm -rf /var/lib/apt/lists/*

# ── 2. Install server Python dependencies (cached layer) ─────────────────────
# Copy requirements first so Docker caches this layer unless reqs change
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# ── 3. Copy the full repository ───────────────────────────────────────────────
COPY . .

# ── 4. Install the guardian inner package (no heavy deps) ─────────────────────
# --no-deps avoids re-installing already-installed packages from reqs
RUN pip install --no-cache-dir -e . --no-deps

# ── 5. Environment variables ──────────────────────────────────────────────────
# OPENAI_API_KEY must be supplied at runtime:
#   docker run -e OPENAI_API_KEY=sk-... guardian-env
ENV PYTHONPATH=/app
ENV PORT=7860
ENV HOST=0.0.0.0
ENV ENABLE_WEB_INTERFACE=true

# ── 6. Create output directories ──────────────────────────────────────────────
RUN mkdir -p /app/outputs/logs /app/outputs/evals /app/guardian/data /app/guardian/checkpoints

# ── 7. Expose port ────────────────────────────────────────────────────────────
EXPOSE 7860

# ── 8. Health check ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# ── 9. Start the FastAPI server ───────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1"]
