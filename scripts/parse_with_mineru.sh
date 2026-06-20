#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="$PROJECT_ROOT/data/papers"
OUTPUT_DIR="$PROJECT_ROOT/data/mineru"
LOG_DIR="$OUTPUT_DIR/_logs"
BACKEND="pipeline"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --input)
      INPUT_DIR="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="${2:-}"
      LOG_DIR="$OUTPUT_DIR/_logs"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if command -v mineru >/dev/null 2>&1; then
  MINERU_CMD=(mineru)
else
  MINERU_CMD=(conda run -n mineru mineru)
fi

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

shopt -s nullglob
pdfs=("$INPUT_DIR"/*.pdf)
if [[ ${#pdfs[@]} -eq 0 ]]; then
  echo "No PDF files found in $INPUT_DIR" >&2
  exit 1
fi

success=0
failed=0
skipped=0

for pdf_path in "${pdfs[@]}"; do
  filename="$(basename "$pdf_path")"
  stem="${filename%.pdf}"
  log_path="$LOG_DIR/$stem.log"

  existing_md="$(find "$OUTPUT_DIR" -type f -name "$stem.md" | head -1)"
  if [[ -n "$existing_md" && "$FORCE" -eq 0 ]]; then
    echo "[skip] $filename -> $existing_md"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "$FORCE" -eq 1 ]]; then
    find "$OUTPUT_DIR" -type f -name "$stem.md" -delete
  fi

  echo "[parse] $filename backend=$BACKEND"
  (
    echo "PDF: $pdf_path"
    echo "Output: $OUTPUT_DIR"
    echo "Backend: $BACKEND"
    echo "Started: $(date -Is)"
    "${MINERU_CMD[@]}" -p "$pdf_path" -o "$OUTPUT_DIR" -b "$BACKEND"
    code=$?
    echo "Finished: $(date -Is)"
    echo "Exit code: $code"
    exit "$code"
  ) >"$log_path" 2>&1

  parsed_md="$(find "$OUTPUT_DIR" -type f -name "$stem.md" | head -1)"
  if [[ -n "$parsed_md" ]]; then
    echo "[ok] $filename -> $parsed_md"
    success=$((success + 1))
  else
    echo "[fail] $filename, see $log_path"
    failed=$((failed + 1))
  fi
done

md_count="$(find "$OUTPUT_DIR" -type f -name '*.md' | wc -l)"

echo
echo "MinerU parse summary"
echo "  success: $success"
echo "  skipped: $skipped"
echo "  failed:  $failed"
echo "  markdown files: $md_count"

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
