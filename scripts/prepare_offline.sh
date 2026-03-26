#!/usr/bin/env bash
# Usage: ./scripts/prepare_offline.sh
#
# Run this ONCE on the development machine (with internet access) before
# delivering the project to the client.
#
# It starts the Ollama service temporarily, pulls both required models
# into ./volumes/ollama/, then stops the service.
# After running, the entire chatbot/ folder (including volumes/) can be
# archived and delivered — the client will not need internet access.

set -euo pipefail

# Ensure .env exists — docker compose requires it for env_file substitution.
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "Copied .env.example → .env (review and update credentials before production use)."
  else
    echo "ERROR: .env file not found. Create it from .env.example before running this script." >&2
    exit 1
  fi
fi

echo "Starting Ollama service..."
docker compose up -d ollama

echo "Waiting for Ollama to be ready..."
timeout_secs=60
elapsed=0
until docker compose exec ollama ollama list > /dev/null 2>&1; do
  sleep 2
  elapsed=$((elapsed + 2))
  if [ "$elapsed" -ge "$timeout_secs" ]; then
    echo "ERROR: Ollama did not become ready within ${timeout_secs}s. Check: docker compose logs ollama" >&2
    docker compose stop ollama
    exit 1
  fi
done

LLM_MODEL="${LLM_MODEL:-qwen2.5:7b-instruct-q4_K_M}"
EMBED_MODEL="${EMBED_MODEL:-multilingual-e5-large}"

echo "Pulling LLM model: $LLM_MODEL"
docker compose exec ollama ollama pull "$LLM_MODEL"

echo "Pulling embedding model: $EMBED_MODEL"
docker compose exec ollama ollama pull "$EMBED_MODEL"

echo "Stopping Ollama service..."
docker compose stop ollama

echo "Pulling chromadb image..."
docker compose pull chromadb

echo "Building backend and frontend images..."
docker compose build backend frontend

echo "Saving Docker images to images.tar..."
docker save \
  ollama/ollama:latest \
  chromadb/chroma:latest \
  chatbot-backend:latest \
  chatbot-frontend:latest \
  -o images.tar

echo ""
echo "Done."
echo "  - Models are stored in ./volumes/ollama/"
echo "  - Docker images are saved to ./images.tar"
echo "Archive the chatbot/ directory (including images.tar and volumes/) and deliver to the client."
echo "On the target server, run: docker load < images.tar  then  docker compose up -d"
