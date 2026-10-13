#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PAPERAGENT_PYTHON:-}" ]]; then
  PYTHON_CMD=("$PAPERAGENT_PYTHON")
elif [[ -x /opt/miniconda3/bin/conda ]]; then
  PYTHON_CMD=(/opt/miniconda3/bin/conda run -n paperagent python)
else
  echo "Set PAPERAGENT_PYTHON to the PaperAgent environment's Python executable." >&2
  exit 1
fi
VECTORSTORE_DIR=""
EMBEDDING_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vectorstore)
      VECTORSTORE_DIR="${2:-}"
      shift 2
      ;;
    --embedding-model)
      EMBEDDING_MODEL="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$PROJECT_ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

VECTORSTORE_DIR="${VECTORSTORE_DIR:-$PROJECT_ROOT/vectorstore}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-$PROJECT_ROOT/models/bge-m3}"

mkdir -p "$VECTORSTORE_DIR"

PAPERAGENT_VECTORSTORE_DIR="$VECTORSTORE_DIR" \
PAPERAGENT_EMBEDDING_MODEL="$EMBEDDING_MODEL" \
  "${PYTHON_CMD[@]}" -m paperagent.retrieval.knowledge_base

echo "  vectorstore: $VECTORSTORE_DIR"
echo "  embedding_model: $EMBEDDING_MODEL"
