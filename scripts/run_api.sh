#!/usr/bin/env bash
set -a
source .env
set +a
source .venv/bin/activate
exec uvicorn api.app:app --reload --port 8000
