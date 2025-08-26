#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="urls.txt"

author_from_url() {
  local url="$1"
  case "$url" in
    *"/bahaullah/"*) echo "Bahá’u’lláh" ;;
    *"/the-bab/"*) echo "The Báb" ;;
    *"/abdul-baha/"*) echo "‘Abdu’l-Bahá" ;;
    *"/prayers/"*) echo "Various" ;;
    *"/shoghi-effendi/"*) echo "Shoghi Effendi" ;;
    *"/the-universal-house-of-justice/"*) echo "The Universal House of Justice" ;;
    *"/compilations/"*) echo "Compilations" ;;
    *) echo "Unknown" ;;
  esac
}

# Titlecase slug (best-effort; we’ll refine later if needed)
title_from_slug() {
  python3 - "$1" <<'PY'
import re,sys
slug=sys.argv[1].replace('-', ' ')
# keep diacritics as-is; just title-case words
print(' '.join(w.capitalize() for w in slug.split()))
PY
}

mkdir -p data/manifests

while IFS= read -r url; do
  [ -z "$url" ] && continue
  base="$(basename "$url" .xhtml)"
  parent="$(dirname "$url")"            # downloads page
  work_id="$base"                       # stable id; for messages it's the date code
  author="$(author_from_url "$url")"
  title="$(title_from_slug "$base")"

  cat > "data/manifests/${work_id}.json" <<JSON
{
  "author": "${author}",
  "work_title": "${title}",
  "work_id": "${work_id}",
  "downloads_page_url": "${parent}/",
  "html_url": "${url}",
  "pdf_url": null,
  "docx_url": null,
  "license_terms_url": "https://www.bahai.org/legal",
  "about_page_url": "${parent}/",
  "hashes": {
    "html": null,
    "pdf": null,
    "docx": null
  },
  "normalized_path": "data/normalized/${work_id}.html",
  "normalized_hash": null
}
JSON
  echo "✓ data/manifests/${work_id}.json"
done < "$INPUT_FILE"
