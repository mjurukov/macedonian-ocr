FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama binary (no systemd, no sudo needed at runtime)
RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64 \
        -o /usr/local/bin/ollama \
    && chmod +x /usr/local/bin/ollama

WORKDIR /app
COPY . .

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
