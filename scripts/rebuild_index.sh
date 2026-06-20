#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PAPERAGENT_PYTHON:-python}"
PROFILE="bge-m3"
VECTORSTORE_DIR=""
EMBEDDING_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
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

if [[ -z "$VECTORSTORE_DIR" || -z "$EMBEDDING_MODEL" ]]; then
  case "$PROFILE" in
    bge-m3)
      VECTORSTORE_DIR="${VECTORSTORE_DIR:-$PROJECT_ROOT/vectorstore}"
      EMBEDDING_MODEL="${EMBEDDING_MODEL:-$PROJECT_ROOT/models/bge-m3}"
      ;;
    current)
      VECTORSTORE_DIR="${VECTORSTORE_DIR:-$PROJECT_ROOT/vectorstore}"
      EMBEDDING_MODEL="${EMBEDDING_MODEL:-$PROJECT_ROOT/models/bge-m3}"
      ;;
    *)
      echo "Unknown profile: $PROFILE" >&2
      exit 2
      ;;
  esac
fi

mkdir -p "$VECTORSTORE_DIR"
rm -f "$VECTORSTORE_DIR/index.faiss" "$VECTORSTORE_DIR/index.pkl"

"$PYTHON_BIN" - <<PY
from paperagent.ingestion import PaperIngestor

PaperIngestor(
    vectorstore_dir=r"$VECTORSTORE_DIR",
    embedding_model=r"$EMBEDDING_MODEL",
    rebuild=True,
)
print("Rebuilt FAISS index")
print("  profile:", "$PROFILE")
print("  vectorstore:", r"$VECTORSTORE_DIR")
print("  embedding_model:", r"$EMBEDDING_MODEL")
PY
