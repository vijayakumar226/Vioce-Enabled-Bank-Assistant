#!/bin/bash
set -e

cd /app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 &

cd /app/frontend
npm run start -- --hostname 0.0.0.0 --port 3000 &

nginx -g "daemon off;"
