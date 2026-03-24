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

echo "Starting Ollama service..."
docker compose up -d ollama

echo "Waiting for Ollama to be ready..."
until docker compose exec ollama ollama list > /dev/null 2>&1; do
  sleep 2
done

LLM_MODEL="${LLM_MODEL:-qwen2.5:7b-instruct-q4_K_M}"
EMBED_MODEL="${EMBED_MODEL:-multilingual-e5-large}"

echo "Pulling LLM model: $LLM_MODEL"
docker compose exec ollama ollama pull "$LLM_MODEL"

echo "Pulling embedding model: $EMBED_MODEL"
docker compose exec ollama ollama pull "$EMBED_MODEL"

echo "Stopping Ollama service..."
docker compose stop ollama

echo ""
echo "Done. Models are stored in ./volumes/ollama/"
echo "Archive the chatbot/ directory and deliver to the client."
