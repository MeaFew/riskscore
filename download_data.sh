#!/usr/bin/env bash
set -euo pipefail
RELEASE_URL="https://github.com/MeaFew/riskscore/releases/download/v1.0-data/data.zip"
DEST_DIR="$(dirname "$0")"
echo "Downloading data for riskscore..."
curl -L -o "${DEST_DIR}/data.zip" "${RELEASE_URL}"
echo "Extracting..."
unzip -o "${DEST_DIR}/data.zip" -d "${DEST_DIR}/data/raw/"
rm "${DEST_DIR}/data.zip"
echo "Done. Run 'make all' to run the single-table pipeline."
echo "For multi-table features, download full Kaggle data (bureau, credit_card, etc.) separately."
