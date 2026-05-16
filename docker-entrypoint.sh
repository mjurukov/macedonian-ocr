#!/bin/bash
set -e

# Start Ollama daemon inside the container (no systemd in Docker)
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
OLLAMA_PID=$!

echo "[entrypoint] Waiting for Ollama..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "[entrypoint] Ollama ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[entrypoint] ERROR: Ollama not responding after 60s."
        kill "$OLLAMA_PID" 2>/dev/null
        exit 1
    fi
    sleep 2
done

exec python3 /app/benchmark_h100.py "$@"
