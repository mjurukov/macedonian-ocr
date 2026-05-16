FROM ollama/ollama:latest

ENV DEBIAN_FRONTEND=noninteractive

# Add Python and curl (curl needed for the entrypoint health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
